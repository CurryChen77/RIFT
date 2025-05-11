#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : carla_runner.py
@Date    : 2023/10/4
"""

import copy
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
from rift.scenario.tools.global_route_planner import GlobalRoutePlanner
from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.ego.base_policy import EgoBasePolicy
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.gym_carla.buffer.collect_buffer import CollectBuffer
from rift.gym_carla.buffer.ego_rollout_buffer import EgoRolloutBuffer
from rift.gym_carla.envs.env_wrapper import VectorWrapper

from rift.ego import EGO_POLICY_LIST
from rift.cbv.planning import CBV_POLICY_LIST
from rift.cbv.recognition import CBV_RECOGNITION_LIST

from rift.scenario.scenario_data_parser import ScenarioDataParser
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.scenario.scenario_data_loader import EvalDataLoader, TrainDataLoader

from rift.scenario.statistics_manager import StatisticsManager
from rift.scenario.tools.route_manipulation import get_latlon_ref
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.util.logger import Logger, setup_logger_dir


MODE_SEED = {
    'train_ego': 1000,
    'train_cbv': 2000,
    'collect_data': 3000,
    'eval': 4000
}


class CarlaRunner:
    def __init__(self, client, traffic_manager, configs: List[Dict]):
        config, ego_config, cbv_config, cbv_recog_config, collect_data_config = configs
        self.config = config
        self.cbv_config = cbv_config
        self.ego_config = ego_config
        self.collect_data_config = collect_data_config
        self.current_map = None

        self.sampling_resolution = 1.0  # used for global route planner
        self.seed = config['seed']
        self.output_dir = config['output_dir']
        self.mode = config['mode']
        self.num_scenario = config['num_scenario']  # default 2
        self.frame_rate = config['frame_rate']
        self.resume = config['resume']
        self.cbv_policy_name = cbv_config['policy_name']
        self.ego_policy_name = ego_config['policy_name']

        # apply settings to carla
        self.client = client
        self.traffic_manager = traffic_manager
        self.world = None
        self.env = None
        self.need_video_render = config['render'] or self.mode == 'collect_data'

        self.env_params = {
            'mode': self.mode,  # the mode of the script
            'num_scenario': self.num_scenario,  # the number of scenarios
            'search_radius': 60,  # the default search radius
            'ego_policy_type': ego_config['policy_type'],  # ego policy type
            'cbv_policy_type': cbv_config['policy_type'],  # cbv policy type
            'ego_obs_type': ego_config['obs_type'],  # ego obs type for ego navigation
            'cbv_obs_type': cbv_config['obs_type'],  # cbv obs type for cbv navigation
            'collect_ego_obs_type': collect_data_config['ego_obs_type'],  # the ego obs type used in collecting data
            'ROOT_DIR': config['ROOT_DIR'],
            'need_video_render': self.need_video_render,  # whether to activate the render
            'signalized_junction': False,  # whether the signal controls the junction
            'warm_up_steps': 4,  # number of ticks after spawning the vehicles
            'img_size': [1280, 720],  # screen size of the camera image
            'acc_range': [-3.0, 3.0],  # continuous acceleration range
            'steer_range': [-0.3, 0.3],  # continuous steering angle range
            'out_lane_thres': 4,  # threshold for out of lane (meter)
            'desired_speed': 8,  # desired speed (m/s)
            'frame_rate': self.frame_rate,  # frame rate of the simulation
        }

        # pass config from scenario to ego
        ego_config['mode'] = self.mode
        ego_config['desired_speed'] = self.env_params['desired_speed']
        ego_config['num_scenario'] = config['num_scenario']
        ego_config['cbv_policy_name'] = cbv_config['policy_name']

        # pass config to cbv recog
        cbv_recog_config['desired_speed'] = self.env_params['desired_speed']

        # pass config from ego to cbv
        cbv_config['ego_policy'] = ego_config['policy_name']
        cbv_config['ego_obs_type'] = ego_config['obs_type']
        cbv_config['need_video_render'] = self.env_params['need_video_render']
        cbv_config['desired_speed'] = self.env_params['desired_speed']

        # pass config to collect_data
        collect_data_config['search_radius'] = self.env_params['search_radius']
        collect_data_config['acc_range'] = self.env_params['acc_range']
        collect_data_config['steer_range'] = self.env_params['steer_range']
        collect_data_config['ego_policy_type'] = ego_config['policy_type']

        CarlaDataProvider.set_desired_speed(self.env_params['desired_speed'])
        CarlaDataProvider.set_frame_rate(self.frame_rate)

        # define logger
        logger_dir = setup_logger_dir(
            self.output_dir,
            self.seed,
            self.mode,
            ego=ego_config['policy_name'],
            cbv=cbv_config['policy_name'],
            cbv_recog=cbv_recog_config['type'],
        )
        self.logger = Logger(logger_dir)

        # prepare parameters
        if self.mode == 'train_ego':
            self.save_freq = ego_config['save_freq']
        elif self.mode == 'train_cbv':
            self.save_freq = cbv_config['save_freq']
        elif self.mode == 'collect_data':
            self.data_path = self.collect_data_config['data_path']
        elif self.mode == 'eval':
            self.logger.log('>> Evaluation Mode, analyzing result', 'yellow')
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

        # define agent and scenario
        self.logger.log('>> Mode: ' + self.mode, color="yellow")
        self.logger.log('>> Ego planning method: ' + ego_config['policy_name'], color="yellow")
        self.logger.log('>> CBV planning method: ' + cbv_config['policy_name'], color="yellow")
        self.logger.log('>> CBV recognition method: ' + cbv_recog_config['type'], color="yellow")
        self.logger.log('>> ' + '-' * 40)

        # define ego policy, scenario policy
        self.ego_policy: EgoBasePolicy = EGO_POLICY_LIST[ego_config['policy_name']](ego_config, logger=self.logger)
        self.cbv_policy: CBVBasePolicy = CBV_POLICY_LIST[cbv_config['policy_name']](cbv_config, logger=self.logger)
        self.cbv_recog_policy: BaseCBVRecog = CBV_RECOGNITION_LIST[cbv_recog_config['type']](cbv_recog_config, logger=self.logger)

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town, reset_settings=False)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.no_rendering_mode = not self.need_video_render
        settings.tile_stream_distance = 800   # used for large map (currently don't support multi-scenareio)
        settings.actor_active_distance = 800  # used for large map (currently don't support multi-scenareio)
        self.world.apply_settings(settings)
        
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world, town)
        CarlaDataProvider.set_traffic_manager_port(self.config['tm_port'])

        # Wait for the world to be ready
        self.world.tick()

        # deterministic mode for reproducing
        self.traffic_manager.set_random_device_seed(self.seed)
        # set the global distance to leading vehicle
        self.traffic_manager.set_global_distance_to_leading_vehicle(8)
        # change the percentage of speed difference, negative value means exceed the speed limit
        self.traffic_manager.global_percentage_speed_difference(-20)
        # Carla Global Route Planner
        CarlaDataProvider.set_global_route_planner(GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution))
        # Carla Map API init
        CarlaDataProvider.set_map_api(CarlaMap(carla_town_name=town))
        # Carla GPS info
        CarlaDataProvider.set_gps_info(get_latlon_ref(self.world))
        # set for cbv recognition
        self.cbv_recog_policy.set_world()

    def train_cbv(self, data_loader: TrainDataLoader, buffer: CBVRolloutBuffer):

        while len(data_loader) > 0:
            
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs = data_loader.sampler()

            # build the env
            self.build_env(sampled_scenario_configs)

            start_time = time.time()

            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            # start loop
            CBV_episode_reward = []
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=False)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=False)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # store to the replay buffer
                buffer.store(data_dict)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])

                scenario_reward = np.concatenate([np.array(list(CBVs_reward.values())) for CBVs_reward in data_dict['CBVs_reward']])
                CBV_episode_reward.append(np.mean(scenario_reward)) if scenario_reward.size != 0 else None

            self.logger.log_route_info(sampled_scenario_configs, np.sum(CBV_episode_reward))
            self.logger.log(f'>> buffer length: {buffer.buffer_pos}')
            self.logger.log(f'>> Scenario Progress: [{data_loader.index}/{data_loader.total}] Time: {time.time() - start_time:.2f}s', 'yellow')
            
            # end up environment
            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')

            # train cbv
            self.cbv_policy.log_episode_reward(np.sum(CBV_episode_reward), episode=data_loader.index)
            self.cbv_policy.train(data_loader.index) if buffer.buffer_full else None

            # save checkpoints
            if data_loader.index % self.save_freq == 0 or len(data_loader) == 0:
                self.cbv_policy.save_model(data_loader.index)

            # save video
            if self.need_video_render:
                self.logger.video_render.save_video(self.current_map)
            
            self.logger.log('>> ' + '-' * 40)
        
        self.cbv_policy.finish()

    def train_ego(self, data_loader: TrainDataLoader, buffer: EgoRolloutBuffer):

        while len(data_loader) > 0:
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs = data_loader.sampler()

            # build the env
            self.build_env(sampled_scenario_configs)

            start_time = time.time()

            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            # start loop
            ego_episode_reward = []
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=False)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=False)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # store to the replay buffer
                buffer.store(data_dict)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])

                ego_episode_reward.append(np.mean(data_dict['ego_reward']))

            self.logger.log_route_info(sampled_scenario_configs, np.sum(ego_episode_reward))
            self.logger.log(f'>> buffer length: {buffer.buffer_pos}')
            self.logger.log(f'>> Scenario Progress: [{data_loader.index}/{data_loader.total}] Time: {time.time() - start_time:.2f}s', 'yellow')
            
            # end up environment
            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')

            # train on-policy ego
            self.ego_policy.log_episode_reward(np.sum(ego_episode_reward), episode=data_loader.index)
            self.ego_policy.train(data_loader.index) if buffer.buffer_full else None

            # save checkpoints
            if data_loader.index % self.save_freq == 0 or len(data_loader) == 0:
                self.ego_policy.save_model(data_loader.index)

            # save video
            if self.need_video_render:
                self.logger.video_render.save_video(self.current_map)
            
            self.logger.log('>> ' + '-' * 40)

        self.ego_policy.finish()

    def eval(self, data_loader: EvalDataLoader):
        
        while len(data_loader) > 0:
            # sample scenarios
            sampled_scenario_configs = data_loader.sampler()

            # build the env
            self.build_env(sampled_scenario_configs)
            
            start_time = time.time()
            
            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=True)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=True)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])
            
            # Save the progress and write the route statistics
            self.statistics_manager.save_progress(data_loader.index, data_loader.total)
            self.statistics_manager.write_statistics()

            # clean up all things
            self.logger.log(">> All scenarios are completed. Cleaning up all actors")
            self.logger.log(f'>> Scenario Evaluation Progress: [{data_loader.index}/{data_loader.total}] Time: {time.time() - start_time:.2f}s', 'yellow')

            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')

            # save video
            if self.need_video_render:
                self.logger.video_render.save_video(self.current_map)
            
            self.logger.log('>> ' + '-' * 40)

        # save global statistics
        self.statistics_manager.compute_global_statistics()
        self.statistics_manager.validate_and_write_statistics()
        self.logger.log(">> Finishing all the evaluation, saving the statistics", color='yellow')

    def collect_data(self, data_loader: TrainDataLoader, buffer: CollectBuffer):

        while len(data_loader) > 0:
            # simulate multiple scenarios in parallel (usually 2 scenarios)
            sampled_scenario_configs = data_loader.sampler()
            
            # build the env
            self.build_env(sampled_scenario_configs)

            ego_obs_list, CBVs_obs_list, info_list = self.env.reset(sampled_scenario_configs)

            # get ego vehicle from scenario
            self.ego_policy.set_ego_and_route(self.env.get_ego_vehicles(), info_list, sampled_scenario_configs)

            # start loop
            while not self.env.all_scenario_done():
                # get action from AV policy and CBV policy (assume using one batch)
                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=False)
                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=False)

                # apply action to env and get obs
                data_dict = self.env.step(ego_actions_dict, CBVs_actions_dict, ego_obs_list, CBVs_obs_list, info_list)

                # store to the replay buffer
                buffer.store(data_dict)

                # for transition
                info_list = copy.deepcopy(data_dict['transition_info'])
                ego_obs_list = copy.deepcopy(data_dict['ego_transition_obs'])
                CBVs_obs_list = copy.deepcopy(data_dict['CBVs_transition_obs'])

            self.logger.log(f'>> dataset length: {len(buffer)}')
            
            # end up environment
            self.env.clean_up()
            # tick to ensure that all destroy commands are executed
            self.world.tick()
            self.logger.log('>> Finish Cleaning', 'yellow')
            self.logger.log('>> ' + '-' * 40)

        # save the data
        self.logger.log(">> Start saving the offline data", 'yellow')
        buffer.save_data()
        self.logger.log(">> Successfully saved the offline data", 'yellow')
        self.logger.log('>> ' + '-' * 40)

    def run(self):
        # get scenario data of different maps, and cluster config according to the town
        config_by_map, total_routes = ScenarioDataParser.scenario_parse(self.config, self.logger)
        # init statistics manager
        self.statistics_manager = StatisticsManager(self.logger.output_dir) if self.mode == 'eval' else None
        # init render
        self.logger.init_video_render(self.env_params) if self.need_video_render else None

        # run with different modes
        if self.mode == 'eval':
            # build eval data loader
            data_loader = EvalDataLoader(config_by_map, self.num_scenario, self.logger)
            # evaluation resume
            self.eval_resume(data_loader)  
            # load the model
            self.ego_policy.load_model()
            self.cbv_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.set_mode('eval')
            # start evaluation
            self.eval(data_loader)

        elif self.mode == 'train_ego':
            # build ego data loader
            data_loader = TrainDataLoader(config_by_map, self.num_scenario, self.logger)
            # ego training resume
            self.train_resume(self.ego_policy, data_loader)
            # build ego buffer
            buffer = EgoRolloutBuffer(self.num_scenario, self.mode, self.ego_config, self.logger)
            self.ego_policy.set_buffer(buffer, total_routes)
            # load model
            self.cbv_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('train')
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.set_mode('eval')
            # start training ego
            self.train_ego(data_loader, buffer)

        elif self.mode == 'train_cbv':
            # cbv data loader
            data_loader = TrainDataLoader(config_by_map, self.num_scenario, self.logger)
            # cbv training resume
            self.train_resume(self.cbv_policy, data_loader)
            # cbv buffer
            buffer = CBVRolloutBuffer(self.num_scenario, self.mode, self.cbv_config, self.logger)
            self.cbv_policy.set_buffer(buffer, total_routes)
            # load model
            self.ego_policy.load_model()
            self.cbv_recog_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.set_mode('train')
            self.cbv_recog_policy.set_mode('train')
            # start training the CBV
            self.train_cbv(data_loader, buffer)

        elif self.mode == 'collect_data':
            # collect data loader
            data_loader = TrainDataLoader(config_by_map, self.num_scenario, self.logger)
            # data collecting resume
            exist, file_name = self.data_resume()
            # general buffer for both agent and scenario
            buffer = CollectBuffer(self.num_scenario, self.mode, self.collect_data_config, self.logger)
            buffer.set_file_name(file_name)
            # load model
            self.ego_policy.load_model()
            self.ego_policy.set_mode('eval')
            self.cbv_policy.load_model()
            self.cbv_policy.set_mode('eval')
            self.cbv_recog_policy.load_model()
            self.cbv_recog_policy.set_mode('eval')
            # start collecting the dataset
            self.collect_data(data_loader, buffer) if not exist else None

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}.")

    def build_env(self, sampled_configs: List[RouteScenarioConfiguration]):
        sampled_maps = {config.town for config in sampled_configs if config.town is not None}
        assert len(sampled_maps) == 1, 'Maps across sampled configs should be the same'
        sampled_map = sampled_maps.pop()
        # set the traffic flow related seed
        self._set_traffic_random_seed(sampled_configs)

        if self.current_map != sampled_map:
            self.current_map = sampled_map  # record the current running map name
            self._init_world(self.current_map)

            # create scenarios within the vectorized wrapper
            self.env = VectorWrapper(
                self.env_params,
                self.statistics_manager,
                self.config,
                self.world,
                self.ego_policy,
                self.cbv_policy,
                self.cbv_recog_policy,
                self.logger,
            )
            self.logger.log(">> Finish scenario initialization")

    def eval_resume(self, data_loader: EvalDataLoader):
        if self.resume:
            resume = data_loader.validate_and_resume()
        else:
            resume = False

        if resume:
            self.logger.log(f'>> Evaluating from {data_loader.index}', color='yellow')
            self.statistics_manager.add_file_records()
        else:
            self.logger.log('>> Evaluating from scratch', color='yellow')
            self.statistics_manager.clear_records()
        
        self.statistics_manager.save_progress(data_loader.index, data_loader.total)
        self.statistics_manager.write_statistics()

    def train_resume(self, policy, data_loader: TrainDataLoader):
        # load policy model
        policy.load_model(self.resume)
        continue_episode = policy.continue_episode
        data_loader.validate_and_resume(continue_episode)

    def data_resume(self) -> Tuple[bool, str]:
        """
        Check if the data file exists, and create the directory if it doesn't exist.

        Returns:
            tuple[bool, str]: A tuple containing a boolean indicating existence
                            and the file path as a string.
        """
        # Construct the file path using pathlib for cleaner syntax
        file_path = Path(self.data_path)
        file_name = file_path / f"{self.ego_policy_name}_{self.cbv_policy_name}.hdf5"

        # Check existence and create directory if necessary
        if file_name.is_file():
            self.logger.log(f'>> exist data on {file_path}', color='red')
            return True, str(file_name)
        else:
            file_path.mkdir(parents=True, exist_ok=True)
            return False, str(file_name)

    def _set_traffic_random_seed(self, sampled_configs: List[RouteScenarioConfiguration]):
        sampled_index = sampled_configs[0].index
        # the traffic random seed related to the sampled config index, the mode
        traffic_random_seed = MODE_SEED[self.mode] + sampled_index
        CarlaDataProvider.set_traffic_random_seed(traffic_random_seed)

    def close(self):
        # save the rest video data of the error running
        if self.need_video_render:
            self.logger.video_render.save_video(self.current_map)
        # close the unfinished env
        if self.env:
            self.env.clean_up()