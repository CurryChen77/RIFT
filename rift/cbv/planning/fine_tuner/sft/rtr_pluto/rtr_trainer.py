import logging
from typing import Dict, List, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from rift.cbv.planning.fine_tuner.rlft.ppo_pluto.ppo_pluto import PPOPlutoModel
from rift.cbv.planning.fine_tuner.sft.utils import global_to_local
from rift.cbv.planning.pluto.controller.pid_controller import PIDController
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
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
        model: PPOPlutoModel,
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
        self.clip_epsilon = model.clip_epsilon
        self.lambda_entropy = model.lambda_entropy

        self.mode_interval = self.radius / self.num_modes

        self.value_criterion = nn.SmoothL1Loss()
        self.ActionDist = torch.distributions.normal.Normal

        if use_drivable_area_loss:
            self.drivable_area_loss = ESDFCollisionLoss()

        self.controller = PIDController(sample_interval=self.frame_rate)

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
        candidate_trajectories = cur_res['trajectory']  # (bs, R, M, T, 6)
        probability = cur_res['probability']  # (bs, R, M)
        bs, R, M = probability.shape

        # only take the valid reference line probability
        cur_r_padding_mask = ~cur_data["reference_line"]["valid_mask"].any(-1)  # (bs, R)
        probability.masked_fill_(cur_r_padding_mask.unsqueeze(-1), -1e8)

        # find the max probability index and corresponding R, M indices
        max_idx = torch.argmax(probability.view(bs, -1), dim=1)
        best_r_idx = max_idx // probability.size(-1)
        best_m_idx = max_idx % probability.size(-1)

        old_mode_index = batch['action_mode_torch']  # (bs, 2), [R_idx, M_idx]

        # ppo loss
        ppo_loss = self.get_ppo_loss(probability, old_mode_index, batch)

        # teacher loss
        teacher_loss = self.get_teacher_loss(candidate_trajectories, probability, best_r_idx, best_m_idx, batch)

        # the weight of PPO loss (follow the default setting of RTR)
        lambda_rl = 5.0

        # # drivable area loss (on best trajectory)
        # # get the best trajectory
        # best_trajectory = candidate_trajectories[torch.arange(bs), best_r_idx, best_m_idx]  # (bs, T, 6)
        # drivable_area_loss = self.get_drivable_area_loss(best_trajectory, cur_data)

        loss = (
            ppo_loss * lambda_rl
            + teacher_loss
            # + drivable_area_loss
        )

        return {
            "loss": loss,
            "ppo_loss": ppo_loss.item(),
            "teacher_loss": teacher_loss.item(),
            # "drivable_area_loss": drivable_area_loss.item(),
        }

    def get_ppo_loss(self, probability, old_mode_index, batch):

        bs, R, M = probability.shape
        log_action_probs = F.log_softmax(probability.view(bs, -1), dim=1).clamp(min=-1e6)  # (bs, R*M)
        log_action_probs = log_action_probs.view(bs, R, M)  # (bs, R, M)
        # log-probability of the chosen action
        cur_log_prob = log_action_probs[torch.arange(bs), old_mode_index[:, 0], old_mode_index[:, 1]]  # (bs,)
        entropy = -torch.sum(torch.exp(log_action_probs) * log_action_probs, dim=(1, 2))
        
        state = batch['state_torch']  # (bs, )
        advantage = batch['advantage_torch'].detach()  # (bs, )
        reward_sum = batch['reward_sum_torch'].detach()  # (bs, )
        old_log_prob = batch['old_log_prob_torch'].detach()  # (bs, )

        value = self.model.value_net(state)
        value_loss = self.value_criterion(value, reward_sum)  # the value criterion is SmoothL1Loss() instead of MSE

        ratio = (cur_log_prob - old_log_prob).exp()
        L1 = advantage * ratio
        L2 = advantage * torch.clamp(ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon)
        surrogate = torch.min(L1, L2).mean()
        actor_loss = -(surrogate + entropy.mean() * self.lambda_entropy)
        return value_loss + actor_loss
    
    def get_teacher_loss(self, candidate_trajectories, probability, best_r_idx, best_m_idx, batch):
        """
        Compute the classification loss.
        args:
            candidate_trajectories: all candidate trajectories (bs, R, M, T, 6)
            probability: probability of each candidate trajectory (bs, R, M)
            best_r_idx: best r index (bs)
            best_m_idx: best m index (bs)
            batch: raw input
        """
        # teacher info
        teacher_infos = batch['teacher_infos']  # (bs, 5)

        flatten_target_label, flatten_logits = self.generate_target_label(
            candidate_trajectories, probability,
            teacher_infos, best_r_idx, best_m_idx
        )  # (bs, R, M)

        teacher_loss = F.cross_entropy(
            flatten_logits,  # (bs, R * M)  (logits of the model)
            flatten_target_label.detach()  # (bs, R * M)  (one-hot target label)
        )

        return teacher_loss

    def generate_target_label(self, candidate_trajectories, probability, teacher_infos, best_r_idx, best_m_idx) -> torch.Tensor:
        """
        Generate the target label for classification loss.
        args:
            candidate_trajectories: all candidate trajectories (bs, R, M, T, 6)
            probability: probability of each candidate trajectory (bs, R, M)
            teacher_infos: teacher infos (bs, 5)
            best_r_idx: best r index (bs)
            best_m_idx: best m index (bs)
        """
        bs, R, M = probability.shape
        # teacher guidance information
        teacher_target_speed = teacher_infos[:, 0]  # (bs,)

        origin = teacher_infos[:, 1:3]
        heading = teacher_infos[:, 3]
        speed = teacher_infos[:, 4]

        local_trajectories = global_to_local(candidate_trajectories, origin, heading, self.frame_rate)  # (bs, R, M, T//frame_rate(8), 2)

        # get the target speed and angle for each candidate trajectory
        candidate_target_speed, candidate_angle = self.controller.batch_control_pid(local_trajectories, speed)  # (bs, R, M)

        # find the closest target speed in candidate_target_speed
        difference = torch.abs(candidate_target_speed - teacher_target_speed[:, None, None])
        min_indices = torch.argmin(difference.view(bs, -1), dim=1)
        r_index = min_indices // M
        m_index = min_indices % M

        target_label = torch.zeros_like(probability)
        # use original best r index, but the m index from teacher guidance
        target_label[torch.arange(bs), best_r_idx, m_index] = 1

        return target_label.reshape(bs, -1), probability.reshape(bs, -1)

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
