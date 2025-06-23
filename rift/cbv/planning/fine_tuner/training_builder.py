from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, cast

from lightning import LightningDataModule, LightningModule, Trainer
from hydra.utils import instantiate

from omegaconf import DictConfig
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    EarlyStopping,
)
from lightning.pytorch.loggers.wandb import WandbLogger

from rift.cbv.planning.fine_tuner.rlft.grpo_pluto.grpo_datamodule import GRPODataModule
from rift.cbv.planning.fine_tuner.rlft.ppo_pluto.ppo_datamodule import PPODataModule
from rift.cbv.planning.fine_tuner.rlft.reinforce_pluto.reinforce_datamodule import ReinforceDataModule
from rift.cbv.planning.fine_tuner.rlft.rift_pluto.rift_datamodule import RIFTDataModule
from rift.cbv.planning.fine_tuner.sft.rs_pluto.rs_datamodule import RewardShapingDataModule
from rift.cbv.planning.fine_tuner.sft.rtr_pluto.rtr_datamodule import RTRDataModule
from rift.cbv.planning.fine_tuner.sft.sft_datamodule import SFTDataModule
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from nuplan_plugin.modeling.torch_module_wrapper import TorchModuleWrapper


@dataclass(frozen=True)
class TrainingEngine:
    """Lightning Module engine dataclass wrapping the lightning trainer, model, and logger."""

    trainer: Trainer  # Module describing NN model, loss, metrics, visualization
    model: LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: LightningDataModule  # Loading data
    training_logger: WandbLogger  # Loading data

    def __repr__(self) -> str:
        """
        :return: String representation of class without expanding the fields.
        """
        return f"<{type(self).__module__}.{type(self).__qualname__} object at {hex(id(self))}>"


def build_lightning_datamodule(
    cfg: DictConfig, buffer: CBVRolloutBuffer
) -> LightningDataModule:
    """
    Build the lightning datamodule from the config.
    :param cfg: Omegaconf dictionary.
    :param buffer: Rollout Buffer for the CBV.
    :return: Instantiated datamodule object.
    """
    DataModule_cfg = cfg.datamodule

    if DataModule_cfg.type == 'sft':
        data_module = SFTDataModule(DataModule_cfg, buffer)
    elif DataModule_cfg.type == 'sft-rtr':
        data_module = RTRDataModule(DataModule_cfg, buffer)
    elif DataModule_cfg.type == 'sft-rs':
        data_module = RewardShapingDataModule(DataModule_cfg, buffer)
    elif DataModule_cfg.type == 'rlft-ppo':
        data_module = PPODataModule(DataModule_cfg, buffer)
    elif DataModule_cfg.type == 'rlft-reinforce':
        data_module = ReinforceDataModule(DataModule_cfg, buffer)
    elif DataModule_cfg.type == 'rlft-grpo':
        data_module = GRPODataModule(DataModule_cfg, buffer)
    elif DataModule_cfg.type == 'rlft-rift':
        data_module = RIFTDataModule(DataModule_cfg, buffer)
    else:
        raise ValueError(f"DataModule type {DataModule_cfg.type} not supported.")

    return data_module


def build_lightning_module(
    cfg: DictConfig, torch_module_wrapper: TorchModuleWrapper
) -> LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    # Create the complete Module
    model = instantiate(
        cfg.custom_trainer,
        model=torch_module_wrapper,
        lr=cfg.lr,
        cl_lr_decay=cfg.cl_lr_decay,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
        frame_rate=cfg.frame_rate,
        trainable_layers=cfg.trainable_layers,
    )

    return cast(LightningModule, model)


def build_training_logger(cfg: DictConfig, carla_episode: int) -> WandbLogger:
    """
    Builds the wandb logger from the config.
    """
    cur_name = f'carla_episode={carla_episode}'
    save_dir = Path(cfg.wandb.dir) / cur_name
    save_dir.mkdir(exist_ok=True, parents=True)

    training_logger = WandbLogger(
        save_dir=save_dir,
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cur_name,
        mode=cfg.wandb.mode,
        log_model=cfg.wandb.log_model,
        resume=cfg.wandb.resume,
    )
    return training_logger


def build_custom_trainer(cfg: DictConfig, dir_path: Path, training_logger: WandbLogger, carla_episode:int) -> Trainer:
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :param dir_path: path to save the model
    :return: built object.
    """
    params = cfg.lightning.trainer.params

    callbacks = [
        ModelCheckpoint(
            dirpath=dir_path,
            filename=f"carla_episode={carla_episode}-epoch={{epoch}}-val_loss={{loss/val_loss:.3f}}",
            auto_insert_metric_name=False,
            monitor=cfg.lightning.trainer.checkpoint.monitor,
            mode=cfg.lightning.trainer.checkpoint.mode,
            save_weights_only=True,
            save_top_k=cfg.lightning.trainer.checkpoint.save_top_k,  # only keep the best model
            save_last=False,  # don't save the last model
        ),
        # EarlyStopping(
        #     monitor=cfg.lightning.trainer.early_stopping.monitor,
        #     patience=cfg.lightning.trainer.early_stopping.patience,
        #     mode=cfg.lightning.trainer.early_stopping.mode,
        # ),
        # RichModelSummary(max_depth=1),  # for debugging
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    
    trainer = Trainer(
        callbacks=callbacks,
        logger=training_logger,
        **params,
    )

    return trainer


def build_training_engine(cfg: DictConfig, dir_path: Path, carla_episode:int, torch_module_wrapper: TorchModuleWrapper, buffer:CBVRolloutBuffer) -> TrainingEngine:
    """
    Build the core lightning modules: Trainer
    :param cfg: omegaconf dictionary
    :param dir_paht: path to save the model
    :param carla_episode: int
    :param torch_module_wrapper: TorchModuleWrapper
    :param buffer: CBVRolloutBuffer
    :return: TrainingEngine
    """
    # logger
    training_logger = build_training_logger(cfg, carla_episode)
    # datamodule
    datamodule = build_lightning_datamodule(cfg, buffer)
    # Lightning Trainer
    trainer = build_custom_trainer(cfg, dir_path, training_logger, carla_episode)
    # Model
    model = build_lightning_module(cfg, torch_module_wrapper)

    engine = TrainingEngine(trainer=trainer, model=model, datamodule=datamodule, training_logger=training_logger)
    
    return engine