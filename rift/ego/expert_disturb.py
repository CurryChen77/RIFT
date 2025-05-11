#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : expert.py
@Date    : 2023/10/22
"""
from typing import Dict

import numpy as np

from rift.ego.base_policy import EgoBasePolicy
from rift.ego.expert.autopilot import AutoPilot


class CarlaExpertDisturbAgent(EgoBasePolicy):
    name = 'expert_disturb'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.throttle_disturb = config['throttle_disturb']
        self.steer_disturb = config['steer_disturb']
        self.brake_disturb = config['brake_disturb']
        self.route = None
        self.planner_list = []
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
            throttle_disturb = np.random.uniform(self.throttle_disturb[0], self.throttle_disturb[1])
            steer_disturb = np.random.uniform(self.steer_disturb[0], self.steer_disturb[1])
            brake_disturb = np.random.uniform(self.brake_disturb[0], self.brake_disturb[1])
            # select the controller that matches the env_id
            control = self.planner_list[info['env_id']].run_step(obs[i])
            # when the expert throttle is not 0, disturb the throttle level
            throttle = min(max(control.throttle + throttle_disturb, 0.01), 1) if control.throttle > 0.01 else control.throttle
            # disturb steer
            steer = min(max(control.steer + steer_disturb, -1), 1)
            # when expert brake is not 0, disturb the brake level
            brake = min(max(control.brake + brake_disturb, 0.01), 1) if control.brake > 0.01 else control.brake
            actions[info['env_id']] = [throttle, steer, brake]

        data = {
            'ego_actions': actions
        }
        return data