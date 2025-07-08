import numpy as np
import cv2
from PIL import Image
import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.datasets.builder import PIPELINES
from mmcv.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class DenseDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        aug_config = input_dict.get("aug_config")
        filename = input_dict["depth_filename"]
        depths = [cv2.imread(name) for name in filename]
        depths = [self.decode(x) for x in depths]
        N = len(depths)
        new_depths = []
        for i in range(N):
            depth = self._img_transform(
                np.uint8(depths[i] * 255) , aug_config,
            )
            depth = np.array(depth).astype(np.float32) / 255 * 1000
            new_depths.append(depth)
        
        gt_depth = []
        for i, depth in enumerate(new_depths):
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                depth_map = cv2.resize(depth, dsize=None, fx=1/downsample, fy=1/downsample)
                mask = (depth_map == 0)
                depth_map = np.clip(depth_map, 0.1, self.max_depth)
                depth_map[mask] = -1
                gt_depth[j].append(depth_map)
        
        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]

        # import matplotlib.pyplot as plt
        # for i, depth in enumerate(input_dict["gt_depth"][0]):
        #     plt.imshow(depth)
        #     plt.colorbar()
        #     plt.savefig(f"vis/depth_hm_{i}.jpg")
        #     plt.close()
        # imgs = input_dict["img"]
        # image = np.concatenate(
        #     [
        #         np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
        #         np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
        #     ],
        #     axis=0,
        # )
        # cv2.imwrite(f"vis/img.jpg", image)

        # for i in range(3):
        #     imgs = gt_depth[i]
        #     image = np.concatenate(
        #         [
        #             np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
        #             np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
        #         ],
        #         axis=0,
        #     )
        #     cv2.imwrite(f"vis/depth_{i}.jpg", image*255)
        return input_dict

    def decode(self, depth):
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        normalized_depth = np.dot(depth[:, :, :3], [65536.0, 256.0, 1.0]) / (256.0 * 256.0 * 256.0 - 1.0)
        return normalized_depth

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        return img


@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict

@PIPELINES.register_module()
class CustomPointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        import torch
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float64) - 1
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        # sort = (ranks + depth / 100.).argsort()
        sort = np.argsort(depth.numpy())
        sort = torch.tensor(sort.copy())
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        import torch
        points_lidar = torch.tensor(results['points']).to(torch.float64)
        imgs = np.stack(results['img'])
        # img_aug_matrix  = results['img_aug_matrix']
        # post_rots = [torch.tensor(single_aug_matrix[:3, :3]).to(torch.float) for single_aug_matrix in img_aug_matrix]
        # post_trans = torch.stack([torch.tensor(single_aug_matrix[:3, 3]).to(torch.float) for single_aug_matrix in img_aug_matrix])
        # import pdb;pdb.set_trace()
        intrins = results['cam_intrinsic']
        depth_map_list = []
        
        for cid in range(len(imgs)):
            # import pdb;pdb.set_trace()
            # lidar2lidarego = torch.tensor(results['lidar2ego']).to(torch.float32)
            # lidarego2global = torch.tensor(results['ego2global']).to(torch.float32)
            # cam2camego = torch.tensor(results['camera2ego'][cid])

            # camego2global = results['camego2global'][cid]

            # cam2img = torch.tensor(intrins[cid]).to(torch.float32)
            
            # lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
            #     lidarego2global.matmul(lidar2lidarego))
            # lidar2img = cam2img.matmul(lidar2cam)
            lidar2img = torch.tensor(results['lidar2img'][cid]).to(torch.float64)
            points_img = points_lidar[:, :3].matmul(
                lidar2img[:3, :3].T.to(torch.float64)) + lidar2img[:3, 3].to(torch.float64).unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            # points_img = points_img.matmul(
            #     post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[1],
                                             imgs.shape[2])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        
        ##################################################################
        i=0
        import cv2
        from PIL import Image
        for image_id in range(imgs.shape[0]):
            i+=1
            image = imgs[image_id]
            gt_depth_image = depth_map[image_id].numpy()
            
            gt_depth_image = np.expand_dims(gt_depth_image,2).repeat(3,2)
            
            #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
            im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
            #convert to mat png
            image[gt_depth_image>0] = im_color[gt_depth_image>0]
            im=Image.fromarray(np.uint8(image))
            #save image
            im.save('vis/visualize_{}.png'.format(i))
        #################################################################

        results['gt_depth_'] = depth_map
        depth_map_ = results["gt_depth"][0]
        depth_map = torch.tensor(depth_map)
        depth_map_ = torch.tensor(depth_map_)

        d1 = depth_map[0][depth_map[0]!=-1]
        d2 = depth_map_[0][depth_map_[0]!=-1]
        import ipdb; ipdb.set_trace()

        return results


@PIPELINES.register_module()
class DepthProbLabelGenerator_ori(object):
    def __init__(
        self,
        max_depth=10,
        min_depth=0.25,
        num_depth=64,
        origin_stride=4,
        strides=[4, 8, 16, 32],
        image_hw=None,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.origin_stride = origin_stride
        self.strides = [stride // origin_stride for stride in strides]
        self.image_hw = np.array(image_hw)
    
    def points2depth(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = self.image_hw
            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.2,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, self.min_depth, self.max_depth)            
            h, w = (int(H / self.origin_stride), int(W / self.origin_stride))
            u = np.floor(U / self.origin_stride).astype(np.int32)
            v = np.floor(V / self.origin_stride).astype(np.int32)
            depth_map = np.ones([h, w], dtype=np.float32) * -1
            depth_map[v, u] = depths
            gt_depth.append(depth_map)
            
        return np.stack(gt_depth)
    
    def __call__(self, input_dict):
        depth = self.points2depth(input_dict)
        depth = np.clip(
            depth,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )
        depth = depth[:, None]
        depth_anchor = np.linspace(
            self.min_depth, self.max_depth, self.num_depth)[:, None, None]
        distance = np.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = np.where(mask, depth_anchor, 0)
        y = depth_gt.sum(axis=1, keepdims=True) - depth_gt
        depth_valid_mask = depth > 0
        depth_prob_gt = np.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        views, _, H, W = depth.shape
        gt = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth_prob_gt, (views, self.num_depth, H//s, s, W//s, s))
            gt_tmp = gt_tmp.sum(axis=-1).sum(axis=3)
            mask_tmp = depth_valid_mask.reshape(views, 1, H//s, s, W//s, s)
            mask_tmp = mask_tmp.sum(axis=-1).sum(axis=3)
            gt_tmp /= np.clip(mask_tmp, a_min=1, a_max=None)
            gt_tmp = gt_tmp.reshape(views, self.num_depth, -1)
            gt_tmp = np.transpose(gt_tmp, (0, 2, 1))
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        return input_dict


@PIPELINES.register_module()
class DepthProbLabelGenerator(object):
    def __init__(
        self,
        max_depth=10,
        min_depth=0.25,
        num_depth=64,
        origin_stride=4,
        strides=[4, 8, 16, 32],
        image_hw=None,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.num_depth = num_depth
        self.origin_stride = origin_stride
        self.strides = [stride // origin_stride for stride in strides]
        self.image_hw = np.array(image_hw)
    
    def points2depth(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = self.image_hw
            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.2,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, self.min_depth, self.max_depth)            
            h, w = (int(H / self.origin_stride), int(W / self.origin_stride))
            u = np.floor(U / self.origin_stride).astype(np.int32)
            v = np.floor(V / self.origin_stride).astype(np.int32)
            depth_map = np.ones([h, w], dtype=np.float32) * -1
            depth_map[v, u] = depths
            gt_depth.append(depth_map)
            
        return np.stack(gt_depth)
    
    def __call__(self, input_dict):
        depth = self.points2depth(input_dict)
        depth = np.clip(
            depth,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )
        depth = depth[:, None]
        depth_anchor = np.linspace(
            self.min_depth, self.max_depth, self.num_depth)[:, None, None]
        distance = np.abs(depth - depth_anchor)
        mask = distance < (depth_anchor[1] - depth_anchor[0])
        depth_gt = np.where(mask, depth_anchor, 0)
        y = depth_gt.sum(axis=1, keepdims=True) - depth_gt
        depth_valid_mask = depth > 0
        depth_prob_gt = np.where(
            (depth_gt != 0) & depth_valid_mask,
            (depth - y) / (depth_gt - y),
            0,
        )
        views, _, H, W = depth.shape
        gt = []
        gt_new = []
        for s in self.strides:
            gt_tmp = np.reshape(
                depth_prob_gt, (views, self.num_depth, H//s, s, W//s, s))
            gt_tmp = gt_tmp.sum(axis=-1).sum(axis=3)
            mask_tmp = depth_valid_mask.reshape(views, 1, H//s, s, W//s, s)
            mask_tmp = mask_tmp.sum(axis=-1).sum(axis=3)
            gt_tmp /= np.clip(mask_tmp, a_min=1, a_max=None)
            gt_new.append(np.clip(gt_tmp, a_min=0.0, a_max=1.0))
            gt_tmp = gt_tmp.reshape(views, self.num_depth, -1)
            gt_tmp = np.transpose(gt_tmp, (0, 2, 1))
            gt.append(gt_tmp)
        gt = np.concatenate(gt, axis=1)
        gt = np.clip(gt, a_min=0.0, a_max=1.0)
        input_dict["depth_prob_gt"] = gt
        input_dict["depth_prob_gt_new"] = gt_new
        return input_dict


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict["lidar2img"])
        )
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 

        for key in [
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'ego_status',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
        
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(
        self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]
    ):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str
