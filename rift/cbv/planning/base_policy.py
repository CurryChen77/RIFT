#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : base_policy.py
@Date    : 2023/10/4
"""


class CBVBasePolicy:
    name = 'base'
    type = 'unlearnable'

    """ This is the template for implementing the policy for a scenario. """
    def __init__(self, config, logger):
        self.config = config
        self.num_scenario = config['num_scenario']
        self._render_data = None
        self.route_planner = None

    def set_buffer(self, buffer, total_routes):
        self.buffer = buffer
        self.total_routes = total_routes

    def set_route_planner(self, route_planner):
        self.route_planner = route_planner

    def train(self, writer, e_i):
        raise NotImplementedError()

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, state, infos, deterministic):
        raise NotImplementedError()
    
    def get_render_data(self, env_id):
        return NotImplementedError()
    
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