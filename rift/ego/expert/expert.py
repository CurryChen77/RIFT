#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : expert.py
@Date    : 2023/10/22
"""
from typing import Dict, List

import numpy as np

from rift.ego.base_policy import EgoBasePolicy
from rift.ego.expert.autopilot import AutoPilot


class CarlaExpertAgent(EgoBasePolicy):
    name = 'expert'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.route = None
        self.planner_list: List[AutoPilot] = []
        for _ in range(config['num_scenario']):
            controller = AutoPilot(self.config, self.logger)  # initialize the controller
            self.planner_list.append(controller)

    def set_ego_and_route(self, ego_vehicles, info, sampled_scenario_configs):
        self.ego_vehicles = ego_vehicles
        for i, ego in enumerate(ego_vehicles):
            gps_route = info[i]['gps_route']  # the gps route
            route = info[i]['route']  # the world coord route
            self.planner_list[i].set_planner(ego, gps_route, route)  # set route for each controller

    def get_action(self, obs, infos, deterministic=False) -> Dict[str, np.ndarray]:
        actions = {}
        for i, info in enumerate(infos):
            # select the controller that matches the env_id
            control = self.planner_list[info['env_id']].run_step(obs[i])
            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            actions[info['env_id']] = [throttle, steer, brake]
        data = {
            'ego_actions': actions
        }
        return data