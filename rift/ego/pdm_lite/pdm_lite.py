#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : pdm_lite.py
@Date    : 2025/01/14
'''
import os
from typing import Dict, List

import numpy as np

from rift.ego.base_policy import EgoBasePolicy
from rift.ego.pdm_lite.autopilot import AutoPilot


class PDM_LITE(EgoBasePolicy):
    name = 'pdm_lite'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.config = config
        target_speed = config['desired_speed']
        frame_rate = config['frame_rate']
        self.logger = logger
        self.route = None
        self.planner_list: List[AutoPilot] = []
        for _ in range(config['num_scenario']):
            controller = AutoPilot(target_speed, frame_rate)  # initialize the controller
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
