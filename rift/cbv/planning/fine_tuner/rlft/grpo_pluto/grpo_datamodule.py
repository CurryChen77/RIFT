#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : grpo_datamodule.py
@Date    : 2025/01/27
'''
import numpy as np
import torch
from typing import Dict, List, Optional
from omegaconf import DictConfig
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.util.torch_util import get_device_name


class GRPOCollate:
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

        # group_advantage
        group_advantage = pad_sequence([torch.from_numpy(CBV_data_dict['CBVs_group_advantage']['advantage']) for CBV_data_dict in batch], batch_first=True)
        group_advantage_mask = pad_sequence([torch.from_numpy(CBV_data_dict['CBVs_group_advantage']['valid_mask']) for CBV_data_dict in batch], batch_first=True)

        # group old prob
        old_group_logits = pad_sequence([torch.from_numpy(CBV_data_dict['CBVs_actions_old_group_logits']['logits']) for CBV_data_dict in batch], batch_first=True)
        old_group_logits_mask = pad_sequence([torch.from_numpy(CBV_data_dict['CBVs_actions_old_group_logits']['valid_mask']) for CBV_data_dict in batch], batch_first=True)

        # group ref prob
        ref_group_logits = pad_sequence([torch.from_numpy(CBV_data_dict['CBVs_actions_ref_group_logits']['logits']) for CBV_data_dict in batch], batch_first=True)
        ref_group_logits_mask = pad_sequence([torch.from_numpy(CBV_data_dict['CBVs_actions_ref_group_logits']['valid_mask']) for CBV_data_dict in batch], batch_first=True)

        output = {
            'cur_pluto_feature_torch': PlutoFeature.collate(cur_to_be_batched_features),
            'group_advantage_torch': group_advantage,              # [bs, max_R, M]
            'group_advantage_mask_torch': group_advantage_mask,    # [bs, max_R, M]
            'old_group_logits_torch': old_group_logits,            # [bs, max_R, M]
            'old_group_logits_mask_torch': old_group_logits_mask,  # [bs, max_R, M]
            'ref_group_logits_torch': ref_group_logits,            # [bs, max_R, M]
            'ref_group_logits_mask_torch': ref_group_logits_mask,  # [bs, max_R, M]
        }

        return output


class GRPODataset(Dataset):
    def __init__(self, cfg: DictConfig, buffer: CBVRolloutBuffer):
        self.buffer = buffer
        self.cfg = cfg
        self.data = {}

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer.sample(idx)   


class GRPODataModule(LightningDataModule):
    def __init__(self,cfg: DictConfig, buffer: CBVRolloutBuffer):
        super().__init__()
        self.buffer = buffer
        self.cfg = cfg

        self.gamma = cfg.gamma

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

        # init the GRPO Dataset
        all_dataset = GRPODataset(self.cfg, self.buffer)

        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = random_split(all_dataset, [self.train_ratio, self.val_ratio])
        else:
            raise ValueError(f'CloseLoop fine-tuning currently only support ["fit"], got ${stage}.')

    def preprocess_buffer(self):
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=GRPOCollate(),
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
            collate_fn=GRPOCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
