from inspect import signature

import torch

from mmcv.utils import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.models.bricks.registry import PLUGIN_LAYERS
from mmcv.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format, deformable_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        freeze_backbone=False,
        freeze_neck=False,
        freeze_perception=False,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        if freeze_backbone:
            self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if freeze_neck:
            self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if freeze_perception:
            self.head.det_head.eval()
            for param in self.head.det_head.parameters():
                param.requires_grad = False
            self.head.map_head.eval()
            for param in self.head.map_head.parameters():
                param.requires_grad = False
         

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        # if return_depth and self.depth_branch is not None:
        #     depths = self.depth_branch(feature_maps, metas.get("focal"))
        # else:
        #     depths = None
        if self.use_deformable_func:
            # feature_maps = feature_maps_format(feature_maps)
            feature_maps = deformable_format(feature_maps)
            
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps[0])
        else:
            depths = None
        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, depth_prob = self.extract_feat(img, True, data)
        model_outs = self.head(feature_maps, data, depth_prob)
        output = self.head.loss(model_outs, data)
        if depth_prob is not None and "depth_prob_gt" in data:
            output["loss_depth"] = self.depth_branch.loss(
                depth_prob, data
            )
        # if depths is not None and "gt_depth" in data:
        #     output["loss_dense_depth"] = self.depth_branch.loss(
        #         depths, data["gt_depth"]
        #     )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps, depth_prob = self.extract_feat(img, True)
        # feature_maps = self.extract_feat(img, False)
        # depth_prob = None

        model_outs = self.head(feature_maps, data, depth_prob)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)

