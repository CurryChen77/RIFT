#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : PlanT.py
@Date    : 2023/10/23
"""
import os
from typing import Dict, List

import numpy as np

from rift.ego.base_policy import EgoBasePolicy
from rift.ego.plant.plant_agent import PlanTAgent


class PlanT(EgoBasePolicy):
    name = 'plant'
    type = 'learnable'

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.route = None
        self.planner_list: List[PlanTAgent] = []
        self.ckpt_path = self.config['model_ckpt_load_path']
        self.logger.log(f">> Loading PlanT ego model from {os.path.basename(self.ckpt_path)}")
        for _ in range(config['num_scenario']):
            controller = PlanTAgent(self.config, self.logger)  # initialize the controller
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
