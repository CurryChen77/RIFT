#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : datamodule.py
@Date    : 2024/12/15
'''
from typing import Dict, List, Optional
from omegaconf import DictConfig
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature


class SFTCollate:
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

        # teacher infos
        batched_teacher_infos = [CBV_data_dict['CBVs_teacher_infos'] for CBV_data_dict in batch]

        output = {
            'cur_pluto_feature_torch': PlutoFeature.collate(cur_to_be_batched_features),
            'teacher_infos': torch.stack(batched_teacher_infos, dim=0),  # (bs, 5)
        }

        return output


class SFTDataset(Dataset):
    def __init__(self, cfg: DictConfig, buffer: CBVRolloutBuffer):
        self.buffer = buffer
        self.cfg = cfg

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer.sample(idx)   

    
class SFTDataModule(LightningDataModule):
    def __init__(self,cfg: DictConfig, buffer: CBVRolloutBuffer):
        super().__init__()
        self.buffer = buffer
        self.cfg = cfg

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
        all_dataset = SFTDataset(self.cfg, self.buffer)

        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = random_split(all_dataset, [self.train_ratio, self.val_ratio])
        else:
            raise ValueError(f'CloseLoop fine-tuning currently only support ["fit"], got ${stage}.')


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=SFTCollate(),
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
            collate_fn=SFTCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
