import torch
import torch.nn as nn
import numpy as np

from mmcv.datasets.dd3d.layers.normalization import Scale
from mmcv.models import Linear, bias_init_with_prob
from mmcv.models.backbones.base_module import Sequential, BaseModule
from mmcv.models.utils import xavier_init
from mmcv.models.bricks.registry import (
    PLUGIN_LAYERS,
)

from ...core.box3d import *
from ..blocks import linear_relu_ln


@PLUGIN_LAYERS.register_module()
class MotionPlanningRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        num_cmd=3,
        use_gru=False,
        gru_cat_tp=False,
    ):
        super(MotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.use_gru = use_gru
        self.gru_cat_tp = gru_cat_tp

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        if self.use_gru:
            if self.gru_cat_tp:
                input_size = 4
            else:
                input_size = 2
            self.plan_reg_branch = nn.GRUCell(
                input_size=input_size,
                hidden_size=self.embed_dims,
            )
            self.output = nn.Linear(embed_dims, 2)
        else:
            self.plan_reg_branch = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(),
                nn.Linear(embed_dims, ego_fut_ts * 2),
            )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
        metas,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        if self.use_gru:
            z = plan_query.flatten(0, 2)
            tp = metas["tp_near"][:, None, None].repeat(1, self.num_cmd, self.ego_fut_mode, 1).flatten(0, 2)
            x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)
            output_wp = []
            for _ in range(self.ego_fut_ts):
                if self.gru_cat_tp:
                    x_in = torch.cat([x, tp.float()], dim=-1)
                else:
                    x_in = x
            
                z = self.plan_reg_branch(x_in, z)
                dx = self.output(z)
                x = dx[:, :2] + x
                output_wp.append(x[:, :2])
            plan_reg = torch.stack(output_wp, dim=1)
            plan_reg = plan_reg.reshape(bs, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        else:
            plan_reg = self.plan_reg_branch(plan_query).reshape(bs, 1, self.num_cmd * self.ego_fut_mode, self.ego_fut_ts, 2)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, plan_reg, planning_status


@PLUGIN_LAYERS.register_module()
class MotionPlanningClsRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=6,
        ego_fut_mode=3,
        num_cmd=3,
        use_gru=False,
        gru_cat_tp=False,
    ):
        super(MotionPlanningClsRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.num_cmd = num_cmd
        self.use_gru = use_gru
        self.gru_cat_tp = gru_cat_tp

        self.motion_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        self.motion_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(embed_dims, 1),
        )
        # if self.use_gru:
        #     if self.gru_cat_tp:
        #         input_size = 4
        #     else:
        #         input_size = 2
        #     self.plan_reg_branch = nn.GRUCell(
        #         input_size=input_size,
        #         hidden_size=self.embed_dims,
        #     )
        #     self.output = nn.Linear(embed_dims, 2)
        # else:
        #     self.plan_reg_branch = nn.Sequential(
        #         nn.Linear(embed_dims, embed_dims),
        #         nn.ReLU(),
        #         nn.Linear(embed_dims, embed_dims),
        #         nn.ReLU(),
        #         nn.Linear(embed_dims, ego_fut_ts * 2),
        #     )
        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 10),
        )

    def init_weight(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.motion_cls_branch[-1].bias, bias_init)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(
        self,
        motion_query,
        plan_query,
        ego_feature,
        ego_anchor_embed,
        metas,
    ):
        bs, num_anchor = motion_query.shape[:2]
        motion_cls = self.motion_cls_branch(motion_query).squeeze(-1)
        motion_reg = self.motion_reg_branch(motion_query).reshape(bs, num_anchor, self.fut_mode, self.fut_ts, 2)
        plan_cls = self.plan_cls_branch(plan_query).squeeze(-1)
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)
        return motion_cls, motion_reg, plan_cls, planning_status