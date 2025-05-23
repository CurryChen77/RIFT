#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : carla_env.py
@Date    : 2023/10/4
"""
import weakref
import numpy as np
import gym
import carla

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.cbv.recognition.base_cbv import BaseCBVRecog
from rift.gym_carla.action import ACTION_TYPE
from rift.gym_carla.action.cbv_action import CBVAction
from rift.gym_carla.action.ego_action import EgoAction
from rift.gym_carla.done.cbv_done import CBVDone
from rift.gym_carla.done.ego_done import EgoDone
from rift.gym_carla.observation import OBSERVATION_LIST

from rift.gym_carla.utils.common import get_nearby_agents, \
    get_ego_min_dis, \
    store_agent_state
from rift.scenario.scenario_manager.route_scenario import RouteScenario
from rift.scenario.scenario_manager.scenario_manager import ScenarioManager
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.gym_carla.visualization.visualize import draw_trajectory
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.scenario.tools.timer import GameTime
from rift.util.logger import Logger


class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """

    def __init__(self, env_params, statistics_manager, world=None, cbv_policy: CBVBasePolicy = None, cbv_recog_policy: BaseCBVRecog = None, logger: Logger=None):
        assert world is not None, "the world passed into CarlaEnv is None"

        self.config = None
        self.world = world
        self.logger = logger

        # Record the time of total steps and resetting steps
        self.total_step = 0
        self.env_id = None
        self.ego_vehicle = None
        self.env_params = env_params
        self.ego_policy_type = env_params['ego_policy_type']
        self.cbv_policy_type = env_params['cbv_policy_type']
        self.mode = env_params['mode']
        self.frame_rate = env_params['frame_rate']

        self.need_video_render = env_params['need_video_render']
        self.camera_sensor = None
        self.ego_collide = None
        self.CBVs_collision_sensor = {}
        self.CBVs_collision = {}

        self.CBVs = {}
        self.gps_route = None
        self.route = None
        self.search_radius = env_params['search_radius']

        # for CBV
        self.cbv_policy = cbv_policy
        self.cbv_recog_policy = cbv_recog_policy
        
        self.statistics_manager = statistics_manager

        # scenario manager
        self.scenario_manager = ScenarioManager(env_params, self.statistics_manager, self.logger)

        # for actions
        self.EgoAction: EgoAction = ACTION_TYPE['ego_action'](env_params['acc_range'], env_params['steer_range'], policy_type=self.ego_policy_type)
        self.CBVAction: CBVAction = ACTION_TYPE['cbv_action'](env_params['acc_range'], env_params['steer_range'], policy_type=self.cbv_policy_type)
        
        # for observations
        self.EgoObs = OBSERVATION_LIST[env_params['ego_obs_type']](env_params, self.EgoAction)
        self.CBVObs = OBSERVATION_LIST[env_params['cbv_obs_type']](env_params, self.CBVAction)
        self.CollectEgoObs = OBSERVATION_LIST[env_params['collect_ego_obs_type']](env_params, self.EgoAction)

        # for route planners
        self.ego_route_planner = self.EgoObs.ego_route_planner
        self.CBV_route_planner = self.CBVObs.CBV_route_planner

        # for rewards
        self.EgoReward = self.EgoObs.ego_reward
        self.CBVReward = self.CBVObs.CBV_reward

        # for dones
        self.EgoDone = EgoDone(self.scenario_manager, self.logger)
        self.CBVDone = CBVDone(self.scenario_manager, self.cbv_recog_policy, self.logger)
    
        # for env wrapper
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']

        # for scenario
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.warm_up_steps = env_params['warm_up_steps']
        self.img_size = env_params['img_size']

        # transfer the CBV route planner to the CBV planning policy and CBV recognition policy
        self.cbv_policy.set_route_planner(self.CBV_route_planner)
        self.cbv_recog_policy.set_route_planner(self.CBV_route_planner)

    def _create_sensors(self):
        if self.need_video_render:
            # camera sensor
            self.camera_img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            # self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))  # for ego view
            self.camera_trans = carla.Transform(carla.Location(x=-6.5, y=0., z=6.5),carla.Rotation(pitch=-30.0))  # for third-person view
            # self.camera_trans = carla.Transform(carla.Location(x=12., y=0., z=20.),carla.Rotation(pitch=-90.0))  # god view
            self.camera_bp = CarlaDataProvider._blueprint_library.find('sensor.camera.rgb')
            # Modify the attributes of the blueprint to set image resolution and field of view.
            self.camera_bp.set_attribute('image_size_x', str(self.img_size[0]))
            self.camera_bp.set_attribute('image_size_y', str(self.img_size[1]))
            self.camera_bp.set_attribute('fov', '120')
            # self.camera_bp.set_attribute('lens_circle_multiplier', str(3.0))
            self.camera_bp.set_attribute('lens_circle_falloff', str(0.0))
            self.camera_bp.set_attribute('motion_blur_intensity', str(0.1))
            self.camera_bp.set_attribute('motion_blur_max_distortion', str(0.1))
            # self.camera_bp.set_attribute('chromatic_aberration_intensity', str(0.5))
            # self.camera_bp.set_attribute('chromatic_aberration_offset', str(0))
            # Set the time in seconds between sensor captures
            self.camera_bp.set_attribute('sensor_tick', str(1.0 / self.frame_rate))

    def _create_scenario(self, config: RouteScenarioConfiguration, env_id: int):
        self.logger.log(f">> Loading scenario: {config.name}")
        # create scenarios according to different types
        scenario = RouteScenario(
            world=self.world,
            config=config,
            env_id=env_id,
            env_params=self.env_params,
            cbv_recog_policy=self.cbv_recog_policy,
            EgoAction=self.EgoAction,
            CBVAction=self.CBVAction,
            logger=self.logger
        )
        # init scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario, config)  # The scenario manager only controls the RouteScenario
        self.EgoDone
        # init for ego controller
        self.route = self.scenario_manager.route_scenario.route  # the global route
        self.gps_route = self.scenario_manager.route_scenario.gps_route  # the global gps route
        self.global_route_waypoints = self.scenario_manager.route_scenario.global_route_waypoints
        # init for reward
        self.EgoReward.set_ego_vehicle(self.ego_vehicle)
        self.CBVReward.set_ego_vehicle(self.ego_vehicle)
        # init for done
        self.EgoDone.set_ego_vehicle(self.ego_vehicle)
        self.CBVDone.set_ego_vehicle(self.ego_vehicle)

    def _run_scenario(self):
        self.scenario_manager.run_scenario()  # init the background vehicle

    def register_CBV_sensor(self, CBV):
        blueprint = CarlaDataProvider._blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=CBV)

        # use weak reference to avoid memory leak
        self_weakref = weakref.ref(self)

        def count_collisions(event):
            self_strongref = self_weakref()
            if self_strongref is not None:
                self_strongref.CBVs_collision[event.actor.id] = {
                    'other_actor_id': event.other_actor.id,
                    'normal_impulse': [event.normal_impulse.x, event.normal_impulse.y, event.normal_impulse.z]
                }

        collision_sensor.listen(lambda event: count_collisions(event))
        self.CBVs_collision_sensor[CBV.id] = collision_sensor
        self.CBVs_collision[CBV.id] = None

    def CBVs_recog(self):
        # Skip CBV calculations if conditions are not met
        if (
            self.cbv_policy_type == 'rule-based' or
            len(self.CBVs) >= self.cbv_recog_policy.max_agent_num or
            self.time_step % 2 != 0 or
            self.time_step <= 25
        ):
            return

        # Select the candidate CBV
        CBVs = self.cbv_recog_policy.get_CBVs(
            self.ego_vehicle,
            set(self.CBVs.keys()),
            self.local_route_waypoints,
            self.rest_route_waypoints,
            self.red_light_state
        )

        for CBV in CBVs:
            assert CBV.id not in self.CBVs, 'The selected CBV should not in the existing CBV list'

            self.CBVs[CBV.id] = CBV
            CBV.set_autopilot(False, CarlaDataProvider.get_traffic_manager_port())  # prepared to be controlled
            self.register_CBV_sensor(CBV)
            CarlaDataProvider.add_CBV(self.ego_vehicle, CBV)

            # update the nearby vehicles around the CBV
            CarlaDataProvider.set_CBV_nearby_agents(self.ego_vehicle, CBV, get_nearby_agents(CBV, self.search_radius))

    def reset(self, config: RouteScenarioConfiguration, env_id: int):
        self.config = config
        self.index = config.index
        self.env_id = env_id

        # create sensor
        self._create_sensors()

        # create RouteScenario, scenario manager, ego_vehicle etc.
        self._create_scenario(config, env_id)
        
        # generate the initial background vehicles
        self._run_scenario()
        
        # atach sensor
        self._attach_sensor()

        # reset the route planner
        self.ego_route_planner.reset(self.ego_vehicle, self.global_route_waypoints)
        self.CBV_route_planner.reset(self.ego_vehicle, self.global_route_waypoints)
        self.local_route_waypoints, self.rest_route_waypoints, self.red_light_state = self.ego_route_planner.run_step()
        self.CBV_route_planner.run_step(self.local_route_waypoints, self.rest_route_waypoints)
        # Update time_steps
        self.time_step = 0

        # applying setting can tick the world and get data from sensors
        # removing this block will cause error: AttributeError: 'NoneType' object has no attribute 'raw_data'
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()

        # update the info in the CarlaDataProvider
        CarlaDataProvider.on_carla_tick()

        # recognize the CBV and update CBV global routes
        self.CBVs_recog()
        self.CBV_route_planner.update_CBV(self.rest_route_waypoints)

        # update agent nearby vehicles
        self._update_agent_nearby_vehicle()

        # update the self.ego_collide status
        self.update_ego_collision()

        # store all actor state
        store_agent_state(self.ego_vehicle)

        # get obs
        ego_obs = self.EgoObs.get_obs(self.ego_vehicle)
        CBVs_obs = self.CBVObs.get_obs(self.ego_vehicle)

        info = self.get_info()
        info.update({
            'route_waypoints': self.global_route_waypoints,  # the global route waypoints
            'gps_route': self.gps_route,  # the global gps route
            'route': self.route,  # the global route
        })

        return ego_obs, CBVs_obs, info

    def _attach_sensor(self):
        if self.need_video_render:
            self_weakref = weakref.ref(self)  # weak reference of self

            # Add camera sensor
            self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
            self.camera_sensor.listen(lambda data, self_ref=self_weakref: get_camera_img(self_ref(), data) if self_ref() else None)

            def get_camera_img(ego_self, data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                ego_self.camera_img = array

    def visualize_actors(self):
        # visualize the past trajectory of all the actor on the map
        if self.need_video_render:
            draw_trajectory(self.world, self.ego_vehicle, self.time_step)
            self.ego_route_planner.vis_route()
            self.CBV_route_planner.vis_route()

    def step_before_tick(self, ego_action, cbv_actions):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp

                # update the actions
                self.scenario_manager.get_update(timestamp, ego_action, cbv_actions)

            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        # route planner
        self.local_route_waypoints, self.rest_route_waypoints, self.red_light_state = self.ego_route_planner.run_step()
        self.CBV_route_planner.run_step(self.local_route_waypoints, self.rest_route_waypoints)

        # update agent nearby vehicles
        self._update_agent_nearby_vehicle()

        # store all actor state
        store_agent_state(self.ego_vehicle)

        # update the running status and check whether terminate or not
        self.scenario_manager.update_running_status(self.CBVs_collision)

        # update the self.ego_collide status
        self.update_ego_collision()

        # data before CBV recog
        data = self.get_data_before_CBV_recog()

        # if CBV collided, then remove it
        self._remove_and_clean_CBV(data)

        # recognize the updated CBV and update CBV global routes
        if self.scenario_manager.running:
            self.CBVs_recog()
            self.CBV_route_planner.update_CBV(self.rest_route_waypoints)

        # data after CBV recog
        data = self.get_data_after_CBV_recog(data)

        # self.visualize_actors()  # visualize the controlled bv and the waypoints in clients side after tick

        # Update time steps
        self.time_step += 1
        self.total_step += 1

        return data

    def get_data_before_CBV_recog(self):
        return {
            'CBVs_next_obs': self.CBVObs.get_obs(self.ego_vehicle),
            'CBVs_reward': self.CBVReward.get_reward(self.CBVs_collision),
            'next_info': self.get_info(),
            'CBVs_terminated': self.CBVDone.terminated(self.CBVs_collision),
            'CBVs_truncated': self.CBVDone.truncated(),
            'CBVs_done': self.CBVDone.done(),
            'CBV_ids': list(self.CBVs.keys())
        }

    def get_data_after_CBV_recog(self, data):
        # get ego training data
        data['ego_next_obs'] = self.EgoObs.get_obs(self.ego_vehicle)
        data['ego_reward'] = self.EgoReward.get_reward(self.local_route_waypoints, self.ego_collide)
        data['ego_terminal'] = self.EgoDone.terminated()
        data['ego_id'] = self.ego_vehicle.id

        # get update transition data
        data['ego_transition_obs'] = self.EgoObs.get_tran_obs(self.ego_vehicle, data['ego_next_obs'])
        data['CBVs_transition_obs'] = self.CBVObs.get_tran_obs(self.ego_vehicle, data['CBVs_next_obs'])
        data['transition_info'] = self.get_info()  # info of updated CBV

        return data

    def get_info(self):
        info = {'env_id': self.env_id}
        # collect data
        info.update(self.get_collection_data()) if self.mode == 'collect_data' else None

        return info

    def get_collection_data(self):
        collect_data = {
            'collect_ego_obs': self.CollectEgoObs.get_obs(self.ego_vehicle),
            'collect_ego_min_dis': get_ego_min_dis(self.ego_vehicle, self.search_radius),
            'collect_ego_collide': float(self.ego_collide),
            'camera_image': self.camera_img
        }
        return collect_data

    def _update_agent_nearby_vehicle(self):
        # update ego nearby vehicles
        CarlaDataProvider.set_ego_nearby_agents(self.ego_vehicle, get_nearby_agents(self.ego_vehicle, self.search_radius))
        # update the nearby vehicles around the CBV
        for CBV in CarlaDataProvider.get_CBVs_by_ego(self.ego_vehicle.id).values():
            CarlaDataProvider.set_CBV_nearby_agents(self.ego_vehicle, CBV, get_nearby_agents(CBV, self.search_radius))

    def update_ego_collision(self):
        self.ego_collide = self.scenario_manager.ego_collision or any(
            collision_event
            and collision_event['other_actor_id'] == self.ego_vehicle.id
            for collision_event in self.CBVs_collision.values()
        )

        self.logger.log(f'>> Ego collide', color='yellow') if self.ego_collide else None

    def _remove_sensor(self):
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.CBVs_collision_sensor:
            # remove the collision sensor that has not been destroyed
            for sensor in self.CBVs_collision_sensor.values():
                if sensor is not None and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            self.CBVs_collision_sensor = {}

    def _remove_CBV_sensor(self, CBV_id):
        sensor = self.CBVs_collision_sensor.pop(CBV_id, None)
        if sensor is not None and sensor.is_alive:
            sensor.stop()
            sensor.destroy()
            self.CBVs_collision.pop(CBV_id)

    def _remove_ego(self):
        if self.ego_vehicle is not None and CarlaDataProvider.actor_id_exists(self.ego_vehicle.id):
            CarlaDataProvider.remove_actor_by_id(self.ego_vehicle.id)
        self.ego_vehicle = None

    def _remove_and_clean_CBV(self, data):
        # remove the truncated CBV from the CBV list and set them free to normal bvs
        CBVs_truncated = data['CBVs_truncated']
        for CBV_id, truncated in CBVs_truncated.items():
            if truncated:
                CBV = self.CBVs.pop(CBV_id, None)
                if CBV is not None:
                    # remove the CBV collision sensor
                    self._remove_CBV_sensor(CBV_id)
                    # remove the truncated CBV from existing CBV lists
                    CBV.set_autopilot(True, CarlaDataProvider.get_traffic_manager_port())  # set the original CBV to normal bvs
                    CarlaDataProvider.pop_CBV_nearby_agents(self.ego_vehicle, CBV)
                    CarlaDataProvider.CBV_back_to_BV(self.ego_vehicle, CBV)

        # clean the terminated CBV
        CBVs_terminated = data['CBVs_terminated']
        for CBV_id, terminated in CBVs_terminated.items():
            if terminated:
                CBV = self.CBVs.pop(CBV_id, None)
                if CBV is not None:
                    # remove sensor
                    self._remove_CBV_sensor(CBV_id)
                    # remove the CBV from the CBV dict
                    CarlaDataProvider.CBV_terminate(self.ego_vehicle, CBV)
                    if CBV_id in CarlaDataProvider.get_CBVs_reach_goal_by_ego(self.ego_vehicle.id).keys():
                        # set the goal reaching CBV free
                        CBV.set_autopilot(True, CarlaDataProvider.get_traffic_manager_port())
                    else:
                        # clean the CBV from the environment
                        if CarlaDataProvider.actor_id_exists(CBV_id):
                            CarlaDataProvider.remove_actor_by_id(CBV_id)
                    CarlaDataProvider.pop_CBV_nearby_agents(self.ego_vehicle, CBV)

    def _reset_variables(self):
        self.CBVs = {}
        self.gps_route = None
        self.route = None
        self.global_route_waypoints = None
        self.local_route_waypoints = None
        self.CBVs_collision = {}

    def clean_up(self):
        # remove temp variables
        self._reset_variables()

        # remove the sensor only when evaluating
        self._remove_sensor()

        # destroy criterion sensors on the ego vehicle
        self.scenario_manager.clean_up()

        # remove the ego vehicle after removing all the sensors
        self._remove_ego()
