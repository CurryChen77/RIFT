from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast

from mmcv.models import Linear, build_activation_layer, build_norm_layer
from mmcv.models.backbones.base_module import Sequential, BaseModule
from mmcv.models.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.models.bricks.drop import build_dropout
from mmcv.models import xavier_init, constant_init
from mmcv.models.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

try:
    from ..ops import deformable_aggregation_func as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        filter_outlier=False,
        min_depth=None,
        max_depth=None,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.filter_outlier = filter_outlier
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        depth_prob,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)

        if self.use_deformable_func:
            points_2d, depth, mask = self.project_points(
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            weights = self._get_weights(
                instance_feature, anchor_embed, metas, mask
            )

            points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
                bs, num_anchor * self.num_pts, -1, 2
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor * self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            if depth_prob is not None:
                depth = depth.permute(0, 2, 3, 1).reshape(
                    bs, num_anchor * self.num_pts, -1, 1
                )
                # normalize depth to [0, depth_prob.shape[-1]-1]
                depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
                depth = depth * (depth_prob.shape[-1] - 1)
                features = DAF(
                    *feature_maps, points_2d, weights, depth_prob, depth
                )
            else:
                features = DAF(*feature_maps, points_2d, weights)
            features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
            features = features.sum(dim=2)
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(
        self, instance_feature, anchor_embed, metas=None, mask=None
    ):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = self.weights_fc(feature)
        if mask is not None and self.filter_outlier:
            mask = mask.permute(0, 2, 1, 3)[..., None, :, None]
            weights = weights.reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
            weights = weights.masked_fill(
                torch.logical_and(~mask, mask.sum(dim=2, keepdim=True) != 0),
                float("-inf"),
            )
        weights = (
            weights.reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        depth = points_2d[..., 2]
        mask = depth > 1e-5
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        mask = mask & (points_2d[..., 0] > 0) & (points_2d[..., 1] > 0)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
            mask = mask & (points_2d[..., 0] < 1) & (points_2d[..., 1] < 1)
        return points_2d, depth, mask

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@PLUGIN_LAYERS.register_module()
class DenseDepthClsNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
        grid_config=None,
        strides=[],
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight
        self.grid_config = grid_config
        self.strides = strides
        dbound = grid_config["depth"]
        self.D = round((dbound[1] - dbound[0]) / dbound[2])

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, self.D, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).softmax(dim=1)
            depths.append(depth)
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.
        for i, depth_pred in enumerate(depth_preds):
            gt_depth = self.get_downsampled_gt_depth(gt_depths[0], self.strides[i])
            depth_pred = depth_pred.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
            fg_mask = gt_depth > 0.0
            gt_depth = gt_depth[fg_mask]
            depth_pred = depth_pred[fg_mask]
            with autocast(enabled=False):
                loss_ = F.binary_cross_entropy(
                    depth_pred,
                    gt_depth,
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum()) * self.loss_weight
                loss = loss + loss_

        return loss

    def get_downsampled_gt_depth(self, gt_depths, feat_down_sample):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // feat_down_sample,
                                   feat_down_sample, W // feat_down_sample,
                                   feat_down_sample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous() 
        gt_depths = gt_depths.view(-1, feat_down_sample * feat_down_sample)
        gt_depths_tmp = torch.where(gt_depths == -1,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // feat_down_sample,
                                   W // feat_down_sample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] - 
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,1:]
        return gt_depths.float()


@PLUGIN_LAYERS.register_module()
class DenseDepthProbNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        min_depth=1.0,
        max_depth=46.0,
        num_depth=45,
        strides=[4, 8, 16, 32],
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_depth = num_depth
        self.strides = strides
        self.loss_weight = loss_weight

        self.depth_layers = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            Linear(embed_dims, num_depth),
        )

    def forward(self, feature_maps):
        depth_prob = self.depth_layers(feature_maps).softmax(dim=-1)
        return depth_prob

    def get_downsample_depth_prob(self, depth):
        depth = depth[0] ## bs, N, H, W
        depth = torch.clip(
            depth,
            min=self.min_depth,
            max=self.max_depth,
        )
        depth = depth[:, :, None]
        depth_anchor = torch.linspace(
            self.min_depth, self.max_depth, self.num_depth)[:, None, None].to(depth)
        distance = torch.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = torch.where(mask, depth_anchor, 0)
        y = depth_gt.sum(dim=2, keepdims=True) - depth_gt
        depth_valid_mask = depth > 0
        depth_prob_gt = torch.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        bs, views, _, H, W = depth.shape
        gt = []
        for s in self.strides:
            gt_tmp = torch.reshape(
                depth_prob_gt, (bs, views, self.num_depth, H//s, s, W//s, s))
            gt_tmp = gt_tmp.sum(dim=-1).sum(dim=4)
            mask_tmp = depth_valid_mask.reshape(bs, views, 1, H//s, s, W//s, s)
            mask_tmp = mask_tmp.sum(dim=-1).sum(dim=4)
            gt_tmp /= torch.clip(mask_tmp, min=1, max=None)
            gt_tmp = gt_tmp.reshape(bs, views, self.num_depth, -1)
            gt_tmp = torch.permute(gt_tmp, (0, 1, 3, 2))
            gt.append(gt_tmp)
        gt = torch.cat(gt, dim=2)
        gt = torch.clip(gt, min=0.0, max=1.0)
        return gt

    def get_downsample_depth_prob_(self, depth):
        depth = depth[0]  # [bs, views, H, W]
        bs, views, H, W = depth.shape
        
        # 限制深度范围并创建有效掩码
        depth = torch.clip(depth, min=self.min_depth, max=self.max_depth)
        valid_mask = (depth > 0) & (depth < self.max_depth)
        
        # 计算深度索引和权重
        bin_width = (self.max_depth - self.min_depth) / (self.num_depth - 1)
        index_float = (depth - self.min_depth) / bin_width
        index0 = torch.floor(index_float).long()
        index1 = index0 + 1
        
        # 处理边界索引
        index0 = torch.clamp(index0, 0, self.num_depth - 1)
        index1 = torch.clamp(index1, 0, self.num_depth - 1)
        
        # 计算权重
        weight1 = index_float - index0.float()
        weight0 = 1.0 - weight1
        weight0 *= valid_mask
        weight1 *= valid_mask

        # 优化关键：避免创建全尺寸深度概率张量
        gt_list = []
        for s in self.strides:
            # 计算下采样后的空间尺寸
            H_s, W_s = H // s, W // s
            
            # 初始化下采样概率张量 [bs, views, num_depth, H_s, W_s]
            prob_down = torch.zeros(
                bs, views, self.num_depth, H_s, W_s,
                device=depth.device, dtype=depth.dtype
            )
            
            # 初始化计数张量 [bs, views, 1, H_s, W_s]
            count_down = torch.zeros(
                bs, views, 1, H_s, W_s,
                device=depth.device, dtype=depth.dtype
            )
            
            # 遍历每个下采样块中的位置
            for i in range(s):
                for j in range(s):
                    # 获取当前偏移位置的深度值
                    depth_ij = depth[:, :, i::s, j::s]
                    valid_ij = valid_mask[:, :, i::s, j::s]
                    
                    # 获取当前偏移位置的索引和权重
                    idx0_ij = index0[:, :, i::s, j::s]
                    idx1_ij = index1[:, :, i::s, j::s]
                    w0_ij = weight0[:, :, i::s, j::s]
                    w1_ij = weight1[:, :, i::s, j::s]
                    
                    # 使用scatter_add_避免创建临时大张量
                    prob_down.scatter_add_(
                        2, idx0_ij.unsqueeze(2), w0_ij.unsqueeze(2)
                    )
                    prob_down.scatter_add_(
                        2, idx1_ij.unsqueeze(2), w1_ij.unsqueeze(2)
                    )
                    
                    # 更新有效计数
                    count_down += valid_ij.unsqueeze(2)
            
            # 归一化处理
            prob_down /= torch.clamp(count_down, min=1.0)
            
            # 重塑为所需格式 [bs, views, (H_s*W_s), num_depth]
            prob_down = prob_down.permute(0, 1, 3, 4, 2).reshape(bs, views, -1, self.num_depth)
            gt_list.append(prob_down)
        
        gt = torch.cat(gt_list, dim=2)
        return torch.clip(gt, 0.0, 1.0)

    def loss(self, depth_prob, data):
        gt_depths = data["depth_prob_gt"].float()
        mask = gt_depths[..., 0] != 1
        with autocast(enabled=False):
            loss_depth = (
                F.binary_cross_entropy(
                    depth_prob[mask], gt_depths[mask]
                )
                * self.loss_weight
            )
        
        return loss_depth


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
