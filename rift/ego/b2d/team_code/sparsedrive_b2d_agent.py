import os
import json
import shutil
import time

import math
from pyquaternion import Quaternion

from PIL import Image
import cv2
import numpy as np
import torch
import pickle

from utils.e2e_recorder import E2ERecorder

import carla
from team_code.pid_controller import PIDController
from team_code.planner import RoutePlanner
from rift.ego.b2d.utils import autonomous_agent
from rift.scenario.tools.carla_data_provider import CarlaDataProvider

from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.datasets.pipelines import Compose

RESIZE_SCALE = 1


lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])

left2right = np.eye(4)
left2right[1,1] = -1


def get_entry_point():
    return 'SparseDriveAgent'

class Clock():
    def __init__(self):
        self.time =  time.time()
        self.verbose = False

    def count(self, tag):
        if self.verbose:
            prev_time = self.time
            self.time = time.time()
            print(tag, self.time - prev_time)
        else:
            pass
    

class SparseDriveAgent(autonomous_agent.AutonomousAgent):
    def __init__(self, config_path, ckpt_path, save_result, logger):
        super().__init__(save_result=save_result)
        # init the config
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.logger = logger
        cfg = Config.fromfile(self.config_path)
        if hasattr(cfg, "plugin"):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, "plugin_dir"):
                    _module_dir = os.path.dirname(os.path.join(cfg.base_dir, cfg.plugin_dir))
                    _module_dir = _module_dir.split("/")
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + "." + m
                    plg_lib = importlib.import_module(_module_path)
  
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
        self.logger.log(f'>> load SparseDrive checkpoint from {self.ckpt_path}', color='yellow')
        self.model.cuda()
        self.model.eval()
        self.test_pipeline = []
        for test_pipeline in cfg.test_pipeline:
            if test_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph','LoadMultiViewImageFromFiles']:
                self.test_pipeline.append(test_pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        self.data_aug_conf = cfg.data_aug_conf

        self.lat_ref, self.lon_ref = 42.0, 2.0
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                [ 0.  ,  0.  ,  1.  ,  0.  ],
                                [ 0.  , -1.  ,  0.  ,  0.  ],
                                [ 0.  , -0.24, -1.19,  1.  ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644,  0.        ,  0.81915204,  0.        ],
                                    [-0.81915204,  0.        ,  0.57357644,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [ 0.22517331, -0.24      , -0.82909407,  1.        ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.        , -0.81915204,  0.        ],
                                    [ 0.81915204,  0.        ,  0.57357644,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [-0.22517331, -0.24      , -0.82909407,  1.        ]]),
        'CAM_BACK':np.array([[-1.00000000e+00,  0.00000000e+00,  1.22464680e-16, 0.00000000e+00],
                            [-1.22464680e-16,  0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                            [ 0.00000000e+00, -1.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                            [-1.97168135e-16, -2.40000000e-01, -1.61000000e+00, 1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.        , -0.93969262,  0.        ],
                                    [ 0.93969262,  0.        , -0.34202014,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [-0.25388956, -0.24      , -0.49288953,  1.        ]]),
        'CAM_BACK_RIGHT':np.array([[-0.34202014,  0.        ,  0.93969262,  0.        ],
                                    [-0.93969262,  0.        , -0.34202014,  0.        ],
                                    [ 0.        , -1.        ,  0.        ,  0.        ],
                                    [ 0.25388956, -0.24      , -0.49288953,  1.        ]])
        }
        self.cam_intrinsic = {
        'CAM_FRONT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                            [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_RIGHT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_LEFT': np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[560.16603057,   0.        , 800.        ],
                            [  0.        , 560.16603057, 450.        ],
                            [  0.        ,   0.        ,   1.        ]]),
        'CAM_BACK_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        }

        self.lidar2img = {}
        for key, value in self.cam_intrinsic.items():
            intrinsic = value * RESIZE_SCALE
            self.cam_intrinsic[key] = intrinsic

            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2cam = self.lidar2cam[key]
            self.lidar2img[key] = viewpad @ lidar2cam.T

        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def setup(self, save_path=None, route_index=None):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        self.pidcontroller = PIDController() 
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        
        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        self.prev_control_cache = []

        # init the visualization recorder
        if self.save_result:
            self.recorder = E2ERecorder(save_path)
            self.save_path = save_path
            self.route_name = f'route_{route_index}'
            self.save_path.mkdir(parents=True, exist_ok=True)

    def _init(self):   
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}
        self.pid_metadata = {}

    def sensors(self):
        W = 1600 * RESIZE_SCALE
        H = 900 * RESIZE_SCALE

        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': W, 'height': H, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': W, 'height': H, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                # lidar
                {   'type': 'sensor.lidar.ray_cast',
                    'x': -0.39, 'y': 0.0, 'z': 1.84,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'range': 85,
                    'rotation_frequency': 10,
                    'channels': 64,
                    'points_per_second': 600000,
                    'dropoff_general_rate': 0.0,
                    'dropoff_intensity_limit': 0.0,
                    'dropoff_zero_intensity': 0.0,
                    'id': 'LIDAR_TOP'
                },
            ]
        if self.save_result:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # change from 20 to 80
        imgs = {}
        for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        results = {}
        results['timestamp'] = self.step / CarlaDataProvider.get_frame_rate()  # remove the fixed 20
        results['img'] = []
        results['lidar2img'] = []
  
        for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            results['img'].append(tick_data['imgs'][cam])
            results['lidar2img'].append(self.lidar2img[cam])
        
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        
        lidar = CarlaDataProvider.get_world().get_actors().filter('*sensor.lidar.ray_cast*')[0]
        world2lidar = lidar.get_transform().get_inverse_matrix()
        world2lidar = lefthand_ego_to_lidar @ world2lidar @ left2right
        lidar2global =  self.invert_pose(world2lidar)
        results['lidar2global'] = lidar2global

        ego_status = np.zeros(10, dtype=np.float32)
        ego_status[:3] = np.array([tick_data['acceleration'][0],-tick_data['acceleration'][1],tick_data['acceleration'][2]])
        ego_status[3:6] = -np.array(tick_data['angular_velocity'])
        ego_status[6:9] = np.array([tick_data['speed'],0,0])
        results["ego_status"] = ego_status
        
        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1
        command_onehot = np.zeros(6)
        command_onehot[command] = 1

        # import ipdb; ipdb.set_trace()
        results['gt_ego_fut_cmd'] = command_onehot
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
        results['tp_near'] = local_command_xy
        results['tp_far'] = local_command_xy

        # ego2world = np.eye(4)
        # ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        # ego2world[0:2,3] = can_bus[0:2]
        # lidar2global = ego2world @ self.lidar2ego
        # results['lidar2global'] = lidar2global

        stacked_img = np.stack(results['img'], axis=-1)
        results['img_shape'] = stacked_img.shape
        results['ori_shape'] = stacked_img.shape
        results['pad_shape'] = stacked_img.shape

        aug_config = self.get_augmentation()
        results["aug_config"] = aug_config
        results = self.test_pipeline(results)
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data):
                    input_data_batch[key] = data.to(self.device)
        output_data_batch = self.model(**input_data_batch)
        out_truck = output_data_batch[0]['img_bbox']['final_planning'].numpy()

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()

        metadata_traj['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)     
        metadata_traj['steer'] = control.steer
        metadata_traj['throttle'] = control.throttle
        metadata_traj['brake'] = control.brake
        metadata_traj['steer_traj'] = float(steer_traj)
        metadata_traj['throttle_traj'] = float(throttle_traj)
        metadata_traj['brake_traj'] = float(brake_traj)
        metadata_traj['plan'] = out_truck.tolist()
        metadata_traj['command'] = command
        self.pid_metadata[self.step] = metadata_traj

        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info

        # save the result
        if self.save_result and self.step % 1 == 0:
            self.save(tick_data)
        self.prev_control = control
        
        if len(self.prev_control_cache)==10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)

        return control

    def save(self, tick_data):
        # save the image
        self.recorder.add_image(tick_data, self.pid_metadata[self.step])

        # meta info
        outfile = open(self.save_path / f'{self.route_name}_meta_info.json', 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / f'{self.route_name}_metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def cleanup(self):
        # save the video
        video_name = f'{self.route_name}_video.mp4'
        self.recorder.save_video(video_name) if self.save_result else None

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    def get_augmentation(self):
        H = 900 * RESIZE_SCALE
        W = 1600 * RESIZE_SCALE
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (
            int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
            - fH
        )
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose