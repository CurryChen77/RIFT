#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : reinforce_datamodule.py
@Date    : 2025/01/27
'''
import numpy as np
import torch
from typing import Dict, List, Optional
from omegaconf import DictConfig
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split

from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature


def compute_return(rewards: torch.tensor, dones: torch.tensor, gamma=0.98):
    assert len(rewards) == len(dones), "Rewards and dones must have the same length"
    
    returns = torch.zeros_like(rewards)
    
    episode_return = 0  # init episode return
    for t in range(len(rewards)-1, -1, -1):
        if dones[t] == 1:
            # if the episode is done, the return is the current reward
            episode_return = rewards[t]
        else:
            # if the episode is not done, the return is the current reward plus the discounted return
            episode_return = rewards[t] + gamma * episode_return
        
        returns[t] = episode_return

    # normalize the returns
    episode_return = (episode_return - episode_return.mean()) / (episode_return.std(dim=0) + 1e-5)
    
    return returns


class RewardShapingCollate:
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

        # return
        return_to_be_batched = [CBV_data_dict['CBVs_return'] for CBV_data_dict in batch]

        output = {
            'cur_pluto_feature_torch': PlutoFeature.collate(cur_to_be_batched_features),
            'return_torch': torch.stack(return_to_be_batched, dim=0),
        }

        return output


class RewardShapingDataset(Dataset):
    def __init__(self, cfg: DictConfig, buffer: CBVRolloutBuffer):
        self.buffer = buffer
        self.cfg = cfg
        self.data = {}

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer.sample(idx)   


class RewardShapingDataModule(LightningDataModule):
    def __init__(self,cfg: DictConfig, buffer: CBVRolloutBuffer):
        super().__init__()
        self.buffer = buffer
        self.cfg = cfg

        self.gamma = cfg.gamma

        self.train_batch_size = cfg.train_batch_size
        self.val_batch_size = cfg.val_batch_size
        self.shuffle = cfg.shuffle
        self.num_workers = cfg.num_workers
        self.pin_memory = cfg.pin_memory
        self.persistent_workers = cfg.persistent_workers
        self.train_ratio = cfg.train_ratio
        self.val_ratio = 1 - self.train_ratio
        self.reward_lambda = cfg.reward_lambda

    def setup(self, stage: Optional[str] = None):
        assert self.buffer.buffer_full, 'The buffer should be full before training'

        # init the Reinforce Dataset
        all_dataset = RewardShapingDataset(self.cfg, self.buffer)

        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = random_split(all_dataset, [self.train_ratio, self.val_ratio])
        else:
            raise ValueError(f'CloseLoop fine-tuning currently only support ["fit"], got ${stage}.')

    def preprocess_buffer(self):
        infraction_rewards = torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_reward'), axis=0))  # (bs, )
        teacher_rewards = torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_teacher_rewards'), axis=0))  # (bs, )
        # weighted reward
        rewards = infraction_rewards + self.reward_lambda * teacher_rewards
        dones = torch.from_numpy(np.stack(self.buffer.get_key_data('CBVs_done'), axis=0)).float()  # (bs, )
        # compute the return
        returns = compute_return(rewards, dones, self.gamma)

        del infraction_rewards, teacher_rewards, rewards, dones

        # add the return to the buffer
        self.buffer.add_extra_data({
            'CBVs_return': returns,
        })

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=RewardShapingCollate(),
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
            collate_fn=RewardShapingCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
