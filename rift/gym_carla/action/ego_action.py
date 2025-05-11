#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : ego_action.py
@Date    : 2024/09/24
"""
import numpy as np

from rift.gym_carla.action.base_action import BaseAction


class EgoAction(BaseAction):
    """
        Ego action usually adopts 3 dimensions output [thr]
    """
    def __init__(self, acc_range, steer_range, policy_type):
        super().__init__(acc_range, steer_range, policy_type)

    def convert_raw_action(self, raw_action):
        # normalize and clip the action
        raw_acc, raw_steer = raw_action
        acc = raw_acc * self.acc_max
        steer = raw_steer * self.steer_max
        acc = max(min(self.acc_max, acc), self.acc_min)
        steer = max(min(self.steer_max, steer), self.steer_min)
        return [acc, steer]

    def convert_action(self, action, allow_reverse=False):
        if self.policy_type == 'rl':
            return self.convert_learnable_action(action, allow_reverse)
        elif self.policy_type == 'plant' or self.policy_type == 'rule-based' or self.policy_type == 'e2e':
            return self.convert_rule_action(action)
        else:
            raise NotImplementedError(f"Unsupported ego policy type: {self.policy_type}.")

    def convert_learnable_action(self, action, allow_reverse):
        acc, steer = self.convert_raw_action(action)  # convert raw network output to acceleration and steering angle

        # Convert acceleration to throttle and brake
        if allow_reverse:
            # allow reverse, convert acc to throttle
            if acc > 0:
                throttle = np.clip(acc / 3, 0, 1)
                brake = 0
                reverse = False
            else:
                reverse = True
                throttle = -np.clip(acc / 3, -1, 0)
                brake = 0
        else:
            reverse = False
            # disable reverse, convert acc to positive throttle and brake
            if acc > 0:
                throttle = np.clip(acc / 3, 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(-acc / 8, 0, 1)

        return [reverse, throttle, steer, brake]

    def convert_rule_action(self, action):
        return [False, action[0], action[1], action[2]]

    def inverse_action(self, reverse, throttle, steer, brake, allow_reverse=False, learnable=False):
        if learnable:
            return self.inverse_learnable_action(reverse, throttle, steer, brake, allow_reverse)
        else:
            return self.inverse_rule_action(throttle, steer, brake)

    def inverse_learnable_action(self, reverse, throttle, steer, brake, allow_reverse=False):
        if allow_reverse:
            acc = throttle * -3.0 if reverse else throttle * 3.0
        else:
            acc = brake * -8.0 if brake != 0 else throttle * 3.0

        return [acc, steer]

    def inverse_rule_action(self, throttle, steer, brake):
        # rule-based agent only got brake value with 1 or 0
        acc = -3.0 if brake >= 0.5 else throttle * 3.0
        return [acc, steer]
