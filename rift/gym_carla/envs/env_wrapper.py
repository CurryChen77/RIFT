#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : env_wrapper.py
@Date    : 2023/10/4
"""

from collections import defaultdict
from typing import List
import gym
import carla
import copy
import numpy as np

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.ego.base_policy import EgoBasePolicy
from rift.gym_carla.envs.carla_env import CarlaEnv
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.scenario.statistics_manager import StatisticsManager
from rift.scenario.tools.timer import GameTime
from rift.util.logger import Logger


class VectorWrapper():
    """ 
        The interface to control a list of environments.
    """
    env_list: List[CarlaEnv]

    def __init__(self, env_params, statistics_manager: StatisticsManager, config, world, ego_policy, cbv_policy, cbv_recog_policy, logger: Logger):
        self.logger = logger
        self.world = world
        self.num_scenario = config['num_scenario']  # default 2
        self.repetitions = config['repetitions']  # default 2
        self.ROOT_DIR = config['ROOT_DIR']
        self.frame_skip = config['frame_skip']
        self.spectator = config['spectator']
        self.need_video_render = env_params['need_video_render']
        self.mode = env_params['mode']

        self.statistics_manager = statistics_manager
        self.ego_policy: EgoBasePolicy = ego_policy
        self.cbv_policy: CBVBasePolicy = cbv_policy
        self.cbv_recog_policy: BaseCBVRecog = cbv_recog_policy

        self.env_list = []
        for i in range(self.num_scenario):
            # each small scenario corresponds to a carla_env create the ObservationWrapper()
            env = carla_env(
                env_params, statistics_manager, world, cbv_policy, cbv_recog_policy, logger
            )
            self.env_list.append(env)

        # flags for env list 
        self.finished_env = [False] * self.num_scenario

    def _initialize_weather(self, config_list: List[RouteScenarioConfiguration]):
        """
        Set the weather
        """
        # Set the appropriate weather conditions
        config_length = len(config_list)
        repeat_indexs = set([config.repetition_index for config in config_list])
        assert len(repeat_indexs) == 1, 'The repetition index should be the same for all the scenarios'
        repeat_index = repeat_indexs.pop()

        if self.repetitions <= config_length:
            config = config_list[repeat_index]
        else:
            config = config_list[repeat_index % config_length]
        self.world.set_weather(config.weather[0][1])

    def get_ego_vehicles(self):
        ego_vehicles = []
        for env in self.env_list:
            if env.ego_vehicle is not None:
                # self.logger.log('>> Ego vehicle is None. Please call reset() first.', 'red')
                # raise Exception()
                ego_vehicles.append(env.ego_vehicle)
        return ego_vehicles

    def reset(self, scenario_data_configs: RouteScenarioConfiguration):
        # create scenarios and ego vehicles
        ego_obs_list = []
        CBVs_obs_list = []
        info_list = []

        # set all the traffic light to green
        CarlaDataProvider.set_all_traffic_light(traffic_light_state=carla.TrafficLightState.Green, timeout=100)

        # set the weather
        self._initialize_weather(scenario_data_configs)

        for s_i in range(len(scenario_data_configs)):
            # for each scenario in the town
            config = scenario_data_configs[s_i]
            ego_obs, CBVs_obs, info = self.env_list[s_i].reset(
                config=config,
                env_id=s_i,  # give each scenario an id
                )
            ego_obs_list.append(ego_obs)
            CBVs_obs_list.append(CBVs_obs)
            info_list.append(info)

        timestamp = self.world.get_snapshot().timestamp
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()  # tick since each small scenario got several warm-up ticks

        if self.spectator:
            transform = CarlaDataProvider.get_first_ego_transform()  # from the first ego vehicle view
            if transform is not None:
                spectator = self.world.get_spectator()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(x=-3, z=50), carla.Rotation(yaw=transform.rotation.yaw, pitch=-80.0)
                ))

        # sometimes not all scenarios are used
        self.finished_env = [False] * self.num_scenario
        for env_id in range(len(scenario_data_configs), self.num_scenario):
            self.finished_env[env_id] = True

        # return obs
        return ego_obs_list, CBVs_obs_list, info_list

    def step(self, ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list):
        """
            ego_actions_dict: Dict['ego_actions': {env_id: ego_action}, 'ego_actions_log_prob': {env_id: ego_action_log_prob}]
            CBVs_actions_dict: Dict['CBVs_actions': {env_id: CBVs_action}, 'CBVs_actions_log_prob': {env_id: CBVs_action_log_prob}]
        """
        # apply action
        for env_id in range(self.num_scenario):
            if not self.finished_env[env_id]:
                # apply the corresponding action by env id
                current_env = self.env_list[env_id]
                current_env.step_before_tick(
                    ego_actions_dict.get('ego_actions')[env_id],
                    CBVs_actions_dict.get('CBVs_actions')[env_id]
                )

                # Render
                if self.need_video_render:
                    # BEV image
                    render_data = self.cbv_policy.get_render_data(env_id)
                    self.logger.video_render.add_BEV_image(current_env.index, render_data)
                    # sensor image
                    sensor_img = copy.deepcopy(current_env.camera_img)
                    self.logger.video_render.add_sensor_image(current_env.index, sensor_img)
        
        if self.need_video_render:
            self.cbv_policy.reset_render_data()

        if self.spectator:
            transform = CarlaDataProvider.get_first_ego_transform()  # from the first ego vehicle view
            if transform is not None:
                spectator = self.world.get_spectator()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(z=50), carla.Rotation(yaw=transform.rotation.yaw, pitch=-80.0)
                ))
        start_time = GameTime.get_time()
        # tick all scenarios
        for _ in range(self.frame_skip):
            self.world.tick()

        # collect new transition of one frame
        transition_data_dict = {
            'ego_transition_obs': [],
            'CBVs_transition_obs': [],
            'transition_info': [],
        }

        data_dict = defaultdict(lambda: list())
        
        timestamp = self.world.get_snapshot().timestamp
        GameTime.on_carla_tick(timestamp)
        # After tick, update all the actors' info
        CarlaDataProvider.on_carla_tick()
        end_time = GameTime.get_time()
        CarlaDataProvider.set_tick_time(end_time - start_time)

        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                current_env = self.env_list[e_i]

                # step after tick
                data = current_env.step_after_tick()

                # check if env is done
                if data['ego_terminal']:
                    self.finished_env[e_i] = True
                    if self.statistics_manager:
                        current_env.scenario_manager.register_statistics()
                else:
                    # append data to the transition data dict, if terminated, then no need for transition
                    for key, value in transition_data_dict.items():
                        value.append(data[key])

                # append the data 
                for key, value in data.items():
                    if key not in transition_data_dict.keys():
                        data_dict[key].append(value)
                # append ego action data
                for key, value in ego_actions_dict.items():
                    data_dict[key].append(value[e_i])
                # append CBVs action data
                for key, value in CBVs_actions_dict.items():
                    data_dict[key].append(value[e_i])

        data_dict.update(transition_data_dict)
        data_dict['ego_obs'] = ego_obs_list
        data_dict['CBVs_obs'] = CBVs_obs_list
        data_dict['info'] = info_list

        return data_dict

    def all_scenario_done(self):
        if np.sum(self.finished_env) == self.num_scenario:
            return True
        else:
            return False

    def clean_up(self):
        # clean up for ego policy
        self.ego_policy.clean_up()

        # clean up for CBV policy
        self.cbv_policy.clean_up()

        # stop sensor objects
        for e_i in range(self.num_scenario):
            self.env_list[e_i].clean_up()

        if self.statistics_manager is not None:
            self.statistics_manager.remove_scenario()

        # clean the CarlaDataProvider
        CarlaDataProvider.clean_up_after_episode()


def carla_env(env_params, statistics_manager, world=None, cbv_policy=None, cbv_recog_policy=None, logger=None):
    return gym.make(
            'carla-v0', 
            env_params=env_params,
            statistics_manager=statistics_manager,
            world=world,
            cbv_policy=cbv_policy,
            cbv_recog_policy=cbv_recog_policy,
            logger=logger,
        )
