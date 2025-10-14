#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : datamodule.py
@Date    : 2024/12/15
'''
import gc
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from omegaconf import DictConfig
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split

from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.util.torch_util import CUDA, get_device_name


def get_advantages_GAE(rewards, undones, values, next_values, unterminated, gamma=0.98, lambda_gae_adv=0.98):
    """
        unterminated: if the CBV collide with an object, then it is terminated
        undone: if the CBV is stuck or collide or max step will cause 'done'
        https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/AgentPPO.py
    """
    advantages = torch.empty_like(values)  # advantage value

    horizon_len = rewards.shape[0]

    advantage = torch.zeros_like(values[0])  # last advantage value by GAE (Generalized Advantage Estimate)

    for t in range(horizon_len - 1, -1, -1):
        delta = rewards[t] + unterminated[t] * gamma * next_values[t] - values[t]
        advantages[t] = advantage = delta + undones[t] * gamma * lambda_gae_adv * advantage
    return advantages


class RTRCollate:
    """Wrapper class that collates together multiple samples into a batch."""

    def __call__(
        self, batch: List[Dict]
    ) -> Dict[str, PlutoFeature]:
        """
        Collate list of [dict] into batch
        :param batch: list of dict to be batched
        :return data already batched
        """
        assert len(batch) > 0, "Batch size has to be greater than 0!"
        # current torch feature
        cur_to_be_batched_features = [CBV_data_dict['CBVs_obs']['raw_pluto_feature'] for CBV_data_dict in batch]

        state_to_be_batched = [CBV_data_dict['CBVs_state'] for CBV_data_dict in batch]
        advantage_to_be_batched = [CBV_data_dict['CBVs_advantage'] for CBV_data_dict in batch]
        reward_sum_to_be_batched = [CBV_data_dict['CBVs_reward_sum'] for CBV_data_dict in batch]
        old_log_prob_to_be_batched = [CBV_data_dict['CBVs_old_log_prob'] for CBV_data_dict in batch]
        action_mode_to_be_batched = [CBV_data_dict['CBVs_action_mode'] for CBV_data_dict in batch]

        # teacher infos
        batched_teacher_infos = [CBV_data_dict['CBVs_teacher_infos'] for CBV_data_dict in batch]

        output = {
            'cur_pluto_feature_torch': PlutoFeature.collate(cur_to_be_batched_features),
            'state_torch': torch.stack(state_to_be_batched, dim=0),
            'advantage_torch': torch.stack(advantage_to_be_batched, dim=0),
            'reward_sum_torch': torch.stack(reward_sum_to_be_batched, dim=0),
            'old_log_prob_torch': torch.stack(old_log_prob_to_be_batched, dim=0),
            'action_mode_torch': torch.stack(action_mode_to_be_batched, dim=0),
            'teacher_infos': torch.stack(batched_teacher_infos, dim=0),  # (bs, 5)
        }

        return output


class RTRDataset(Dataset):
    def __init__(self, cfg: DictConfig, buffer: CBVRolloutBuffer):
        self.buffer = buffer
        self.cfg = cfg
        self.data = {}

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer.sample(1)        
    
    def sample(self, batch_size):
        return self.buffer.sample(batch_size)


class RTRDataModule(LightningDataModule):
    def __init__(self,cfg: DictConfig, buffer: CBVRolloutBuffer):
        super().__init__()
        self.buffer = buffer
        self.cfg = cfg

        self.gamma = cfg.gamma
        self.lambda_gae_adv = cfg.lambda_gae_adv

        self.device = get_device_name()

        self.train_batch_size = cfg.train_batch_size
        self.val_batch_size = cfg.val_batch_size
        self.shuffle = cfg.shuffle
        self.num_workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.persistent_workers = cfg.persistent_workers
        self.train_ratio = cfg.train_ratio
        self.val_ratio = 1 - self.train_ratio

    def setup(self, stage: Optional[str] = None):
        assert self.buffer.buffer_full, 'The buffer should be full before training'

        # init the RL Dataset
        all_dataset = RTRDataset(self.cfg, self.buffer)

        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = random_split(all_dataset, [self.train_ratio, self.val_ratio])
        else:
            raise ValueError(f'CloseLoop fine-tuning currently only support ["fit"], got ${stage}.')

    def preprocess_buffer(self, model: torch.nn.Module):
        model = CUDA(model)  # put the model on the GPU

        rewards = torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_reward'), axis=0))  # (bs, )
        undones = 1.0 - torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_done'), axis=0)).float()  # (bs, )
        unterminated = 1.0 - torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_terminated'), axis=0)).float()
        old_log_prob = torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_actions_old_log_prob'), axis=0))  # (bs, )
        action_mode = torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_actions_mode'), axis=0))  # (bs, )

        chunk_size = self.train_batch_size

        cur_to_be_batched_features = [obs['raw_pluto_feature'] for obs in self.buffer.get_key_data('CBVs_obs')]
        cur_states = []
        cur_values = []
        for i in range(0, len(cur_to_be_batched_features), chunk_size):
            chunk_features = cur_to_be_batched_features[i:i+chunk_size]
            chunk_torch = PlutoFeature.collate(chunk_features).to_device(self.device)
            # forwarding
            with torch.no_grad():
                chunk_output = model(chunk_torch.data)
                chunk_value = model.value_net(chunk_output['hidden'])
            cur_states.append(chunk_output['hidden'].cpu())
            cur_values.append(chunk_value.cpu())

        next_to_be_batched_features = [next_obs['raw_pluto_feature'] for next_obs in self.buffer.get_key_data('CBVs_next_obs')]
        next_values = []
        for i in range(0, len(next_to_be_batched_features), chunk_size):
            chunk_features = next_to_be_batched_features[i:i+chunk_size]
            chunk_torch = PlutoFeature.collate(chunk_features).to_device(self.device)
            # forwarding
            with torch.no_grad():
                chunk_output = model(chunk_torch.data)
                chunk_value = model.value_net(chunk_output['hidden'])
            next_values.append(chunk_value.cpu())

        cur_state = torch.cat(cur_states, dim=0)
        value = torch.cat(cur_values, dim=0)
        next_value = torch.cat(next_values, dim=0)

        # compute the advantages
        advantages = get_advantages_GAE(rewards, undones, value, next_value, unterminated, gamma=self.gamma, lambda_gae_adv=self.lambda_gae_adv)
        reward_sums = advantages + value

        del (
            rewards, undones, unterminated,
            cur_to_be_batched_features, next_to_be_batched_features,
            value, next_value
        )

        advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)

        self.buffer.add_extra_data({
            'CBVs_state': cur_state.cpu(),
            'CBVs_advantage': advantages.cpu(),
            'CBVs_reward_sum': reward_sums.cpu(),
            'CBVs_old_log_prob': old_log_prob.cpu(),
            'CBVs_action_mode': action_mode.cpu()
        })

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=RTRCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=RTRCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
