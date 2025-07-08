import torch

import numpy as np
from numpy import random
import mmcv
from mmcv.datasets.builder import PIPELINES
from PIL import Image


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                np.uint8(imgs[i]), aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                # results["cam_intrinsic"][i][:3, :3] = (
                #     mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                # )

        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

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

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix


@PIPELINES.register_module()
class BBoxRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], angle
            )
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

@PIPELINES.register_module()
class BBoxMapRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (
                results["lidar2img"][view] @ rot_mat_inv
            )
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(
                results["gt_bboxes_3d"], angle
            )
        
        if "map_geoms" in results:
            map_geoms_rotate = {}
            for label, geom_list in results["map_geoms"].items():
                map_geoms_rotate[label] = []
                for geom in geom_list:
                    geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                    map_geoms_rotate[label].append(geom_rotate)
            results["map_geoms"] = map_geoms_rotate
        
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

@PIPELINES.register_module()
class MapRotation(object):
    def __init__(self, rot_range=[-0.3925, 0.3925]):
        self.rot_range = rot_range

    def __call__(self, results):
        angle = np.random.uniform(*self.rot_range)
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        map_geoms_rotate = {}
        for label, geom_list in results["map_geoms"].items():
            map_geoms_rotate[label] = []
            for geom in geom_list:
                geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                map_geoms_rotate[label].append(geom_rotate)
        results["map_geoms"] = map_geoms_rotate

        return results


import os
import numpy as np
import matplotlib.pyplot as plt
import shapely
import shapely.affinity
from shapely.geometry import LineString, Polygon
from mmcv.utils import mkdir_or_exist

@PIPELINES.register_module()
class MapRotationVis(object):
    def __init__(self, rot_range=[-0.3925, 0.3925], vis_dir=None):
        self.rot_range = rot_range
        self.vis_dir = './vis'
        if vis_dir is not None:
            mkdir_or_exist(vis_dir)
    
    def plot_geoms(self, geoms_dict, filepath):
        """绘制并保存几何图形"""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        for label, geom_list in geoms_dict.items():
            for geom in geom_list:
                if isinstance(geom, LineString):
                    x, y = geom.xy
                    ax.plot(x, y, label=label)
                elif isinstance(geom, Polygon):
                    x, y = geom.exterior.xy
                    ax.fill(x, y, alpha=0.5, label=label)
        
        ax.axis('equal')
        ax.grid(True)
        ax.legend()
        plt.savefig(filepath)
        plt.close()
    
    def __call__(self, results):
        angle = np.random.uniform(*self.rot_range)
        angle = 90/180*np.pi
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)
        
        # 可视化旋转前的几何
        if self.vis_dir is not None:
            sample_token = results.get('sample_token', 'unknown')
            before_path = os.path.join(self.vis_dir, f'{sample_token}_before_rot.png')
            self.plot_geoms(results["map_geoms"], before_path)
        
        # 执行旋转
        map_geoms_rotate = {}
        for label, geom_list in results["map_geoms"].items():
            map_geoms_rotate[label] = []
            for geom in geom_list:
                geom_rotate = shapely.affinity.rotate(geom, angle, origin=(0, 0), use_radians=True)
                map_geoms_rotate[label].append(geom_rotate)
        results["map_geoms"] = map_geoms_rotate
        
        # 可视化旋转后的几何
        if self.vis_dir is not None:
            after_path = os.path.join(self.vis_dir, f'{sample_token}_after_rot_{np.rad2deg(angle):.1f}deg.png')
            self.plot_geoms(results["map_geoms"], after_path)
        return results

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mmcv.utils import mkdir_or_exist

@PIPELINES.register_module()
class BBoxRotationVis(object):
    def __init__(self, vis_dir=None):
        self.vis_dir = "vis/"
        if vis_dir is not None:
            mkdir_or_exist(vis_dir)
    
    def plot_boxes(self, boxes, filepath, title=""):
        """绘制3D边界框的俯视图(xy平面)"""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        for box in boxes:
            # 提取中心坐标、长宽和朝向角
            x, y, z = box[:3]
            length, width = box[3], box[4]
            yaw = box[6]
            
            # 计算四个角点
            half_l, half_w = length/2, width/2
            corners = np.array([
                [-half_l, -half_w],
                [ half_l, -half_w],
                [ half_l,  half_w],
                [-half_l,  half_w]
            ])
            
            # 旋转角点
            rot_mat = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])
            rotated_corners = corners @ rot_mat.T + np.array([x, y])
            
            # 绘制边界框
            rect = plt.Polygon(rotated_corners, closed=True, 
                              fill=False, linewidth=2, edgecolor='r')
            ax.add_patch(rect)
            
            # 绘制朝向箭头
            front = rotated_corners[1] - rotated_corners[0]
            front = front / np.linalg.norm(front)
            ax.arrow(x, y, front[0]*2, front[1]*2, 
                    head_width=0.5, head_length=0.7, fc='b', ec='b')
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title(title)
        ax.grid(True)
        ax.axis('equal')
        
        # 设置合理的坐标范围
        if len(boxes) > 0:
            centers = boxes[:, :2]
            max_extent = np.max(boxes[:, [3,4]]) * 1.5
            x_min, y_min = np.min(centers, axis=0) - max_extent
            x_max, y_max = np.max(centers, axis=0) + max_extent
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.savefig(filepath)
        plt.close()
    
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        angle = 90/180*np.pi
        # 可视化旋转前的边界框
        if self.vis_dir is not None and "gt_bboxes_3d" in results:
            sample_token = results.get('sample_token', 'unknown')
            before_path = os.path.join(self.vis_dir, f'{sample_token}_bbox_before_rot.png')
            self.plot_boxes(results["gt_bboxes_3d"], before_path, 
                          f"Before Rotation (Angle: {np.rad2deg(angle):.1f}°)")
        
        # 执行旋转变换
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)
        
        # 变换相机参数
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv
        
        # 变换全局坐标
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        
        # 变换3D边界框
        if "gt_bboxes_3d" in results:
            original_boxes = results["gt_bboxes_3d"].copy()
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)
            
            # 可视化旋转后的边界框
            if self.vis_dir is not None:
                after_path = os.path.join(self.vis_dir, f'{sample_token}_bbox_after_rot.png')
                self.plot_boxes(results["gt_bboxes_3d"], after_path,
                              f"After Rotation (Angle: {np.rad2deg(angle):.1f}°)")
                
                # 可选：在同一图中绘制前后对比
                compare_path = os.path.join(self.vis_dir, f'{sample_token}_bbox_compare.png')
                plt.figure(figsize=(10, 10))
                ax = plt.gca()
                
                # 绘制原始框(红色)
                for box in original_boxes:
                    self._draw_single_box(ax, box, 'r', 'Original')
                
                # 绘制旋转后框(蓝色)
                for box in results["gt_bboxes_3d"]:
                    self._draw_single_box(ax, box, 'b', 'Rotated')
                
                ax.legend()
                ax.set_title(f"Rotation Comparison ({np.rad2deg(angle):.1f}°)")
                ax.grid(True)
                ax.axis('equal')
                plt.savefig(compare_path)
                plt.close()
        import ipdb; ipdb.set_trace()
        return results
    
    def _draw_single_box(self, ax, box, color, label):
        """辅助函数：绘制单个边界框"""
        x, y = box[:2]
        length, width = box[3], box[4]
        yaw = box[6]
        
        half_l, half_w = length/2, width/2
        corners = np.array([
            [-half_l, -half_w],
            [ half_l, -half_w],
            [ half_l,  half_w],
            [-half_l,  half_w]
        ])
        
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        rotated_corners = corners @ rot_mat.T + np.array([x, y])
        
        rect = plt.Polygon(rotated_corners, closed=True, 
                          fill=False, linewidth=2, edgecolor=color, label=label)
        ax.add_patch(rect)
        
        front = rotated_corners[1] - rotated_corners[0]
        front = front / np.linalg.norm(front)
        ax.arrow(x, y, front[0]*2, front[1]*2, 
                head_width=0.5, head_length=0.7, fc=color, ec=color)
    
    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array(
            [[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]]
        )
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d