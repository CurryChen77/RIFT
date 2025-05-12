#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : e2e_agent.py
@Date    : 2025/05/8
'''
from typing import Dict, List

import numpy as np
from rift.ego.base_policy import EgoBasePolicy
from rift.ego.b2d.team_code.vad_b2b_agent import VadAgent
from team_code.uniad_b2d_agent import UniadAgent
from rift.ego.b2d.utils.agent_wrapper import AgentWrapper, validate_sensor_configuration
from rift.ego.b2d.utils.watchdog import Watchdog
from rift.scenario.tools.route_scenario_configuration import RouteScenarioConfiguration
from rift.util.logger import Logger


class E2E_Agent(EgoBasePolicy):
    name = 'e2e'
    type = 'learnable'

    def __init__(self, config, logger: Logger):
        self.config = config
        self.logger = logger
        self.route = None
        self.planner_list: List[AgentWrapper] = []

    def set_ego_and_route(self, ego_vehicles, info, sampled_scenario_configs: List[RouteScenarioConfiguration]):
        self.ego_vehicles = ego_vehicles
        agent_watchdog = Watchdog(50)
        agent_watchdog.start()
        for i, (ego, config) in enumerate(zip(ego_vehicles, sampled_scenario_configs)):
            wrapper_planner = self.planner_list[i]
            planner = wrapper_planner.agent()
            gps_route = info[i]['gps_route']  # the gps route
            route = info[i]['route']  # the world coord route
            # AV planner setup
            town = config.town
            route_index = config.index

            save_path = self.logger.output_dir / str(town) / f'route_{route_index}'
            planner.set_global_plan(gps_route, route, ego)
            planner.setup(save_path=save_path)
            # AV planner wrapper setup
            wrapper_planner.setup_sensors(ego)

        agent_watchdog.stop()

    def get_action(self, obs, infos, deterministic=False) -> Dict[str, np.ndarray]:
        actions = {}
        for i, info in enumerate(infos):
            # select the planner that matches the env_id
            wrapper_planner = self.planner_list[info['env_id']]
            control = wrapper_planner()
            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            actions[info['env_id']] = [throttle, steer, brake]
        data = {
            'ego_actions': actions
        }
        return data
    
    def clean_up(self):
        for planner in self.planner_list:
            planner.cleanup()
            planner.agent().reset()
        self.logger.log(f'>> {self.name} agent clean up.')


class VAD(E2E_Agent):
    name = 'vad'
    type = 'learnable'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        # init the AV method        
        for _ in range(config['num_scenario']):
            # init the AV planner
            planner = VadAgent(
                config_path=config['config_path'],
                ckpt_path=config['ckpt_path'],
                save_result=config['save_result'],
                logger=self.logger
                )
            # validate the sensor configuration
            sensors = planner.sensors()
            track = planner.track 
            validate_sensor_configuration(sensors, track, 'SENSORS')
            # store the wrapper planner
            planner_wrapper = AgentWrapper(planner)
            self.planner_list.append(planner_wrapper)

class UniAD(E2E_Agent):
    name = 'uniad'
    type = 'learnable'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        # init the AV method        
        for _ in range(config['num_scenario']):
            # init the AV planner
            planner = UniadAgent(
                config_path=config['config_path'],
                ckpt_path=config['ckpt_path'],
                save_result=config['save_result'],
                logger=self.logger
                )
            # validate the sensor configuration
            sensors = planner.sensors()
            track = planner.track 
            validate_sensor_configuration(sensors, track, 'SENSORS')
            # store the wrapper planner
            planner_wrapper = AgentWrapper(planner)
            self.planner_list.append(planner_wrapper)

