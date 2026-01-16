#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : cbv_rollout_buffer.py
@Date    : 2024/09/21
"""
from collections import defaultdict, deque
import random
from typing import Set
import numpy as np
import torch

from rift.gym_carla.buffer.base_buffer import BaseBuffer


class CBVRolloutBuffer(BaseBuffer):
    name = 'CBVRolloutBuffer'

    """
        This buffer designed for rl-based on-policy methods
    """
    def __init__(
            self,
            num_scenario,
            mode,
            cbv_config,
            logger=None):
        super().__init__(num_scenario, mode, logger)
        assert self.mode == 'train_cbv', f'Only initialize {self.name} when training the rl-based onpolicy cbv agent'
        
        self.buffer_capacity = cbv_config['buffer_capacity']
        self.data_keys = cbv_config['data_keys']

        self.reset_buffer()

    def reset_buffer(self):
        self.buffer_pos = 0
        self.buffer_full = False

        # Use defaultdict to create deque for each data type
        self.buffer_data = {
            key: deque(maxlen=self.buffer_capacity) for key in self.data_keys
        }

        # Temporary buffer to store intermediate data
        self.temp_buffer = {key: defaultdict(list) for key in self.buffer_data}

    def process_data_dict(self, data_dict):
        """
            the processed_data_dict contains continuous ego trajectory data
        """
        # initialize
        processed_data_dict = {key: [] for key in self.buffer_data.keys()}

        scenario_lengths = set(len(data) for key, data in data_dict.items() if key in self.buffer_data.keys())
        assert len(scenario_lengths) == 1, 'all the data in the data dict should have same length'
        scenario_length = scenario_lengths.pop()

        CBV_ids_list = [CBV_ids for CBV_ids in data_dict['CBV_ids']]

        for i in range(scenario_length):
            for CBV_id in CBV_ids_list[i]:
                for key, value in self.temp_buffer.items():
                    value[CBV_id].append(data_dict[key][i][CBV_id])
                
                if data_dict['CBVs_done'][i][CBV_id]:
                    # the trajectory of ego is completed, pop the whole trajectory
                    for key, value in processed_data_dict.items():
                        value.extend(self.temp_buffer[key].pop(CBV_id))  # pop out specific whole trajectory data

        data_lengths = set(len(data) for data in processed_data_dict.values())
        assert len(data_lengths) == 1, 'the data in the processed data dict should have same length'
        data_length = data_lengths.pop()

        return processed_data_dict, data_length

    def store(self, data_dict):
        processed_data_dict, data_length = self.process_data_dict(data_dict)

        if data_length > 5:  # ignore too short ego trajectory
            if self.buffer_pos + data_length >= self.buffer_capacity:
                # the buffer can just hold part of the trajectory data
                for i in range(data_length):
                    if self.buffer_pos < self.buffer_capacity:
                        for key, data in self.buffer_data.items():
                            data.append(processed_data_dict[key][i])
                        self.buffer_pos += 1
                    else:
                        break
                self.buffer_full = True
            else:
                # the buffer still can hold the whole trajectory data
                for key, data in self.buffer_data.items():
                    data.extend(processed_data_dict[key])
                self.buffer_pos += data_length

    def get_all_np_data(self):
        '''
        the np array data for traditional rl-based on-policy methods
        '''
        assert self.buffer_pos == self.buffer_capacity, 'only get the data when the buffer is full'
        batch_dict = {}
        for key, deque_data in self.buffer_data.items():
            batch_dict[key] = np.stack(deque_data).reshape(self.buffer_capacity, -1)

        return batch_dict
    
    def add_extra_data(self, data_dict: dict):
        """
        Add data to the buffer
        """
        assert self.buffer_full, 'only add data when the buffer is full'
        assert all(len(v) == self.buffer_capacity for v in data_dict.values())
        # update the buffer data
        self.buffer_data.update(data_dict)
    
    def get_key_data(self, key: str):
        """
        Get the data from the buffer
        """
        assert self.buffer_full, 'only get the data when the buffer is full'
        return self.buffer_data[key]
    
    def sample(self, idx):
        '''
        sample the data from the buffer
        '''
        assert self.buffer_full, 'only sample the data when the buffer is full'

        # deterministic fetch when an index (or indices) is provided
        indices = idx if isinstance(idx, (list, tuple)) else [idx]
        assert all(0 <= i < self.buffer_capacity for i in indices)

        sampled_data = {
            key: [deque_data[i] for i in indices] if len(indices) > 1 else deque_data[indices[0]]
            for key, deque_data in self.buffer_data.items()
        }
        return sampled_data
