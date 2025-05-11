#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : behavior.py
@Date    : 2023/10/4
"""
from typing import Dict

import numpy as np

from rift.ego.base_policy import EgoBasePolicy
from agents.navigation.behavior_agent import BehaviorAgent


class CarlaBehaviorAgent(EgoBasePolicy):
    name = 'behavior'
    type = 'unlearnable'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.route = None
        self.planner_list = []
        self.behavior = config['behavior']

    def set_ego_and_route(self, ego_vehicles, info, sampled_scenario_configs):
        self.ego_vehicles = ego_vehicles
        self.planner_list = []
        for e_i in range(len(ego_vehicles)):
            controller = BehaviorAgent(self.ego_vehicles[e_i], behavior=self.behavior)
            dest_waypoint = info[e_i]['route_waypoints'][-1]  # the destination of the ego vehicle
            location = dest_waypoint.transform.location
            controller.set_destination(location)  # set route for each controller
            self.planner_list.append(controller)

    def get_action(self, obs, infos, deterministic=False) -> Dict[str, np.ndarray]:
        actions = {}
        for info in infos:
            controller = self.planner_list[info['env_id']]
            # the waypoint list in rift and carla's behavior agent is different
            # for the behavior agent, the goal may be reached (no more waypoints to chase), but rift still got waypoints
            if controller.done():
                throttle = 0.1
                steer = 0
                brake = 0
            else:
                # select the controller that matches the env_id
                control = controller.run_step(debug=False)
                throttle = control.throttle
                steer = control.steer
                brake = control.brake
            actions[info['env_id']] = [throttle, steer, brake]

        data = {
            'ego_actions': actions
        }
        return data
