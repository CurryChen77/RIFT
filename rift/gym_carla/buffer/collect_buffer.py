#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : cbv_rollout_buffer.py
@Date    : 2024/09/21
"""
import os
from pathlib import Path
import imageio
import h5py
import numpy as np

from rift.gym_carla.buffer.base_buffer import BaseBuffer
from rift.gym_carla.action.ego_action import EgoAction


class CollectBuffer(BaseBuffer):
    name = 'CollectBuffer'

    """
        This buffer designed for data collection
    """
    def __init__(
            self,
            num_scenario,
            mode,
            collect_data_config,
            logger=None):
        super().__init__(num_scenario, mode, logger)
        assert self.mode == 'collect_data', f'this buffer design for data collection'

        self.buffer_capacity = collect_data_config['buffer_capacity']
        self.obs_shape = tuple(collect_data_config['ego_obs_shape'])
        self.action_dim = collect_data_config['ego_action_dim']
        self.state_dim = collect_data_config['ego_obs_dim']
        self.ego_policy_type = collect_data_config['ego_policy_type']
        self.search_radius = collect_data_config['search_radius']
        self.EgoAction = EgoAction(collect_data_config['acc_range'], collect_data_config['steer_range'], self.ego_policy_type)
        self.ego_id_set = set()

        # for transformer fine-tuning data
        self.file_name = None
        self.image_id = 0
        self.image_data_path = None
        self.image_base_name = None

        self.reset_buffer()

    def reset_buffer(self):
        self.buffer_pos = 0
        self.buffer_full = False
        self.image_id = 0

        self.buffer_data = {
            'collect_ego_actions': np.zeros((self.buffer_capacity, self.action_dim), dtype=np.float32),
            'collect_ego_obs': np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32),
            'collect_ego_next_obs': np.zeros((self.buffer_capacity, *self.obs_shape), dtype=np.float32),
            'collect_ego_reward': np.zeros(self.buffer_capacity, dtype=np.float32),
            'collect_ego_terminal': np.zeros(self.buffer_capacity, dtype=np.float32),
            'collect_ego_min_dis': np.zeros(self.buffer_capacity, dtype=np.float32),
            'collect_ego_collide': np.zeros(self.buffer_capacity, dtype=np.float32),
            'camera_image_path': [],
        }

        self.temp_buffer = {key: {} for key in self.buffer_data}
        self.ego_id_set = set()

    def process_ego_action(self, raw_action):
        if self.ego_policy_type == 'rl':
            # convert ego action network's raw output to (acceleration, steering)
            ego_action = self.EgoAction.convert_raw_action(raw_action)
        elif self.ego_policy_type == 'plant' or self.ego_policy_type == 'rule-based' or self.ego_policy_type == 'e2e':
            # convert the rule-based ego action (throttle, steering) to (acceleration, steering)
            ego_action = self.EgoAction.inverse_rule_action(raw_action[0], raw_action[1], raw_action[2])
        else:
            raise NotImplementedError(f"Unsupported ego policy type: {self.ego_policy_type}.")
        return ego_action

    def pre_process_data(self, data_dict):
        processed_data_dict = {
            'collect_ego_actions': [np.array(self.process_ego_action(ego_action), dtype=np.float32) for ego_action in data_dict['ego_actions']],
            'collect_ego_obs': [np.array(data['collect_ego_obs'], dtype=np.float32) for data in data_dict['info']],
            'collect_ego_next_obs': [np.array(data['collect_ego_obs'], dtype=np.float32) for data in data_dict['next_info']],
            'collect_ego_min_dis': [np.array(data['collect_ego_min_dis'], dtype=np.float32) for data in data_dict['next_info']],
            'collect_ego_collide': [np.array(data['collect_ego_collide'], dtype=np.float32) for data in data_dict['next_info']],
            'collect_ego_reward': np.array(data_dict['ego_reward'], dtype=np.float32),
            'collect_ego_terminal': np.array(data_dict['ego_terminal'], dtype=np.float32),
        }
        return processed_data_dict

    def process_data_dict(self, data_dict):
        """
            the processed_data_dict contains continuous ego trajectory data
        """
        # initialize
        processed_data_dict = {key: [] for key in self.buffer_data.keys()}

        scenario_length = len(data_dict['ego_obs'])

        pre_processed_data_dict = self.pre_process_data(data_dict)

        for i in range(scenario_length):
            ego_id = data_dict['ego_id'][i]
            if ego_id not in self.ego_id_set:
                # initialize for new ego_id
                self.ego_id_set.add(ego_id)
                for key, value in self.temp_buffer.items():
                    value[ego_id] = []

            # transfer necessary data from the processed data dict to the temp buffer
            for key, value in pre_processed_data_dict.items():
                self.temp_buffer[key][ego_id].append(value[i])

            # add the camera image info
            image_path = self.save_image(data_dict['info'][i]['camera_image'], self.image_id)
            self.temp_buffer['camera_image_path'][ego_id].append(image_path)
            self.image_id += 1

            if data_dict['ego_terminal'][i]:
                # the trajectory of ego is completed, pop the whole trajectory
                for key, value in processed_data_dict.items():
                    value.extend(self.temp_buffer[key].pop(ego_id))  # pop out specific whole trajectory data

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
                            # key code: store the data_dict to the self.buffer_data
                            if key.startswith('collect'):
                                data[self.buffer_pos] = processed_data_dict[key][i]
                            else:
                                data.append(processed_data_dict[key][i])
                        self.buffer_pos += 1
                    else:
                        break
                self.buffer_full = True
            else:
                # the buffer still can hold the whole trajectory data
                for key, data in self.buffer_data.items():
                    if key.startswith('collect'):
                        data[self.buffer_pos:self.buffer_pos + data_length] = processed_data_dict[key]
                    else:
                        data.extend(processed_data_dict[key])
                self.buffer_pos += data_length

    def set_file_name(self, file_name):
        self.file_name = Path(file_name)
        # Town03_expert_standard
        self.image_base_name = f"{self.file_name.parent.name}_{self.file_name.stem}"
        self.image_data_path = self.file_name.parent / 'images'
        self.image_data_path.mkdir(parents=True, exist_ok=True)

    def save_image(self, image, image_id):
        image_path = f"{self.image_data_path}/{self.image_base_name}_{image_id}.png"
        imageio.imwrite(image_path, image.astype(np.uint8))
        return image_path
    
    def save_data(self):
        assert self.buffer_pos == self.buffer_capacity, 'only get the data when the buffer is full'

        with h5py.File(self.file_name, 'w') as file:
            str_dt = h5py.special_dtype(vlen=str)
            file.attrs['length'] = int(self.buffer_pos)
            file.attrs.update({'action_dim': self.action_dim, 'obs_shape': self.obs_shape})
            for key, data in self.buffer_data.items():
                key_prefix = key.split("_")[0]
                if key_prefix == 'collect':
                    file.create_dataset(key, dtype=np.float32, data=data, compression='gzip')
                elif key_prefix == 'camera':
                    file.create_dataset(key, dtype=str_dt, data=data, compression='gzip')
                else:
                    raise NotImplementedError(f"Unsupported prefix type: {key_prefix}.")
