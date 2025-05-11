#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : base_policy.py
@Date    : 2023/10/4
"""


class EgoBasePolicy:
    name = 'base'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.ego_vehicles = None
        self.config = config

    def set_ego_and_route(self, ego_vehicles, info, sampled_scenario_configs):
        self.ego_vehicles = ego_vehicles
    
    def set_buffer(self, buffer, total_routes):
        self.buffer = buffer
        self.total_routes = total_routes

    def train(self, e_i):
        raise NotImplementedError()

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, state, infos, deterministic):
        raise NotImplementedError()

    def log_episode_reward(self, episode_reward, episode):
        pass

    def load_model(self, resume=True):
        pass

    def save_model(self, episode):
        pass
    
    def clean_up(self):
        pass

    def finish(self):
        pass
