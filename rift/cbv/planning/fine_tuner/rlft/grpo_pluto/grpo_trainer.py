import logging
from typing import Dict, List, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.cbv.planning.pluto.model.pluto_model import PlanningModel
from nuplan_plugin.modeling.types import (
    FeaturesType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from rift.cbv.planning.pluto.optim.warmup_cos_lr import WarmupCosLR

from rift.cbv.planning.pluto.model.loss.esdf_collision_loss import ESDFCollisionLoss

logger = logging.getLogger(__name__)


class LightningTrainer(L.LightningModule):
    def __init__(
        self,
        model: PlanningModel,
        lr,
        cl_lr_decay,
        weight_decay,
        epochs,
        warmup_epochs,
        frame_rate: int,
        trainable_layers: List[str],
        use_drivable_area_loss=True,
        use_regulate_yaw=True,
        objective_aggregate_mode: str = "mean",
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.cl_lr_decay = cl_lr_decay
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.future_steps = model.future_steps
        self.frame_rate = int(frame_rate)
        self.trainable_layers = trainable_layers

        self.use_drivable_area_loss = use_drivable_area_loss
        self.use_regulate_yaw = use_regulate_yaw

        self.radius = model.radius
        self.num_modes = model.num_modes
        self.mode_interval = self.radius / self.num_modes

        if use_drivable_area_loss:
            self.drivable_area_loss = ESDFCollisionLoss()

        self.freeze_parameters(trainable_layers)

    def freeze_parameters(self, trainable_layers=["planning_decoder.pi_head"]):
        # freeze all param
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze specific layer
        for layer_name in trainable_layers:
            # get the module
            layer = dict(self.model.named_modules()).get(layer_name, None)
            if layer is None:
                raise ValueError(f"Layer {layer_name} not found in the model.")
            for param in layer.parameters():
                param.requires_grad = True

    # def on_train_start(self):
    #     print(">> Parameter `requires_grad` states:")
    #     for name, param in self.model.named_parameters():
    #         print(f">> {name}: requires_grad={param.requires_grad}")

    def _step(
        self, batch: Dict[str, PlutoFeature], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        # current input output pair
        cur_data = batch['cur_pluto_feature_torch'].data
        cur_res = self.forward(cur_data)

        # compute the training objective
        losses = self._compute_objectives(cur_res, cur_data, batch)

        self._log_step(losses["loss"], losses, prefix)

        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, cur_res, cur_data, batch) -> Dict[str, torch.Tensor]:
        probability = cur_res['probability']  # (bs, R, M)

        # grpo loss
        grpo_loss = self.get_grpo_loss(probability, cur_data, batch)

        # drivable area loss (on best trajectory)
        # drivable_area_loss = self.get_drivable_area_loss(best_trajectory, cur_data)

        loss = (
            grpo_loss
            # + drivable_area_loss
        )

        return {
            "loss": loss,
            "grpo_loss": grpo_loss.item(),
            # "drivable_area_loss": drivable_area_loss.item(),
        }

    def get_grpo_loss(self, probability, cur_data, batch):
        '''
        Compute the grpo loss.
        '''
        bs, R, M = probability.shape  # (max_R*M=G)
        # only take the valid reference line probability
        cur_r_padding_mask = ~cur_data["reference_line"]["valid_mask"].any(-1)  # (bs, max_R)

        valid_mask = batch['group_advantage_mask_torch']  # (bs, max_R, M)
        r_padding_mask = ~valid_mask.any(-1)
        assert torch.equal(cur_r_padding_mask, r_padding_mask)
        
        # current probability and log probability
        probability.masked_fill_(cur_r_padding_mask.unsqueeze(-1), -1e8)
        log_action_prob = F.log_softmax(probability.view(bs, -1), dim=1)  # (bs, max_R * M)

        # Apply masks to old logits and compute log prob
        old_group_logits = batch['old_group_logits_torch']  # (bs, max_R, M)
        old_group_logits.masked_fill_(cur_r_padding_mask.unsqueeze(-1), -1e8)
        old_group_log_prob = F.log_softmax(old_group_logits.view(bs, -1), dim=1)  # (bs, max_R * M)
        
        # reshape group_advantage to (bs, G)
        group_advantage = batch['group_advantage_torch'].view(bs, -1)  # (bs, G)

        # reference model logits
        ref_group_logits = batch['ref_group_logits_torch']  # (bs, max_R, M)
        ref_group_logits.masked_fill_(cur_r_padding_mask.unsqueeze(-1), -1e8)
        ref_group_prob = F.softmax(ref_group_logits.view(bs, -1), dim=1)  # (bs, max_R * M)

        # GRPO KL divergence
        kl = F.kl_div(
            input=log_action_prob,
            target=ref_group_prob,
            reduction="none",
            log_target=False,
        )
        
        # Compute the truncated objective
        ratio = torch.exp(log_action_prob - old_group_log_prob)
        L1 = group_advantage * ratio
        L2 = group_advantage * torch.clamp(ratio, 0.8, 1.2)

        objective = torch.min(L1, L2) - 0.2 * kl

        # Compute surrogate loss considering only valid positions
        valid_mask_flat = valid_mask.view(bs, -1)  # (bs, G)
        valid_objective = objective[valid_mask_flat]
        if valid_objective.numel() == 0:
            surrogate = torch.tensor(0.0, device=probability.device)
        else:
            surrogate = valid_objective.mean()

        grpo_loss = -surrogate

        return grpo_loss

    def get_drivable_area_loss(self, best_trajectory: torch.Tensor, data):
        """
        Compute the drivable area loss.
        args:
            data: raw input data feature
            best_trajectory: most probable trajectory (bs, T, 6)
        """

        if self.use_drivable_area_loss:
            drivable_area_loss = self.drivable_area_loss(
                best_trajectory, data["cost_maps"][:, :, :, 0].float()  # cost_maps.shape (bs, H, W, 1)
            )
        else:
            drivable_area_loss = best_trajectory.new_zeros(1)
        
        return drivable_area_loss

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def training_step(
        self, batch: Dict, batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Dict, batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Dict, batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                
                # only contain the trainable param
                if not param.requires_grad:
                    continue

                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        
        # only contain the param requires grad
        param_dict = {param_name: param for param_name, param in self.named_parameters() if param.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=self.lr * self.cl_lr_decay,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
