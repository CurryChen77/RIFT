#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : PlanT_agent.py
@Date    : 2023/10/22
"""
import time
import cv2
from pathlib import Path
import warnings
from PIL import Image, ImageDraw, ImageOps
import torch
import math

from rift.ego.utils.coordinate_utils import inverse_conversion_2d
from rift.ego.utils.explainability_utils import *
from rift.util.torch_util import CUDA

from rift.ego.plant.data_agent_boxes import DataAgent
from rift.ego.plant.dataset import generate_batch, split_large_BB
from rift.ego.plant.lit_module import LitHFLM

from rift.ego.expert.nav_planner import RoutePlanner_new as RoutePlanner


warnings.filterwarnings("ignore", category=DeprecationWarning)


class PlanTAgent(DataAgent):
    def __init__(self, config, logger):
        super().__init__(config=config)
        self.config = config
        self.logger = logger
        self.exec_or_inter = config['exec_or_inter']
        self.viz_attn_map = config['viz_attn_map']

        self.save_mask = []
        self.save_topdowns = []
        self.timings_run_step = []
        self.timings_forward_model = []

        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        # exec_or_inter is used for the interpretability metric
        # exec is the model that executes the actions in carla
        # inter is the model that obtains attention scores and a ranking of the vehicles importance
        LOAD_CKPT_PATH = None
        if self.exec_or_inter is not None:
            if self.exec_or_inter == 'exec':
                LOAD_CKPT_PATH = self.config['exec_model_ckpt_load_path']
        else:
            LOAD_CKPT_PATH = self.config['model_ckpt_load_path']

        if Path(LOAD_CKPT_PATH).suffix == '.ckpt':
            self.net = CUDA(LitHFLM.load_from_checkpoint(LOAD_CKPT_PATH, strict=True, cfg=self.config))
        else:
            raise Exception(f'Unknown model type: {Path(LOAD_CKPT_PATH).suffix}')
        self.net.eval()

    def set_planner(self, ego_vehicle, global_plan_gps, global_plan_world_coord):
        super().set_planner(ego_vehicle, global_plan_gps, global_plan_world_coord)
        self._route_planner = RoutePlanner(7.5, 50.0)
        self._route_planner.set_route(self._global_plan, True)

    def tick(self, input_data):
        result = super().tick(input_data)

        waypoint_route = self._route_planner.run_step(result['gps'])

        if len(waypoint_route) > 2:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[2]
        elif len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
            next_target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]
            next_target_point, _ = waypoint_route[0]

        ego_target_point = inverse_conversion_2d(target_point, result['gps'], result['compass'])
        result['target_point'] = tuple(ego_target_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, viz_route=None):
        # The input data contains [speed, imu(yaw angle), gps(x, y location)] sometimes include 'rgb_back', 'sem_back'

        self.step += 1

        # needed for traffic_light_hazard
        _ = super()._get_brake(stop_sign_hazard=0, vehicle_hazard=0, walker_hazard=0)
        tick_data = self.tick(input_data)
        # label_raw contains [vehicle information, route information]
        # pos from the GT data instead of UnscentedKalmanFilter
        label_raw = super().get_bev_boxes(input_data=input_data, pos=input_data['gps'], viz_route=viz_route)

        if self.exec_or_inter == 'exec' or self.exec_or_inter is None:
            self.control = self._get_control(label_raw, tick_data)
        
        inital_frames_delay = 2
        if self.step < inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            
        return self.control

    def _get_control(self, label_raw, input_data):
        
        gt_velocity = CUDA(torch.FloatTensor([input_data['speed']]))

        # input_data contains [speed, imu(yaw angle), gps(x, y location)]
        x, y, _, tp, light = self.get_input_batch(label_raw, input_data)
    
        _, _, pred_wp, attn_map = self.net(x, y, target_point=tp, light_hazard=light)

        steer, throttle, brake = self.net.model.control_pid(pred_wp[:1], gt_velocity)

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        if brake:
            steer *= self.steer_damping

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        viz_trigger = ((self.step % 20 == 0) and self.cfg['viz'])
        if viz_trigger and self.step > 2:
            create_BEV(label_raw, light, tp, pred_wp)

        if self.viz_attn_map:
            attn_vector = get_attn_norm_vehicles(self.cfg['attention_score'], self.data_car, attn_map)
            keep_vehicle_ids, attn_indices, keep_vehicle_attn = get_vehicleID_from_attn_scores(self.data, self.data_car, self.cfg['topk'], attn_vector)
            draw_attention_bb_in_carla(self._world, keep_vehicle_ids, keep_vehicle_attn)

        return control
    
    def get_input_batch(self, label_raw, input_data):
        sample = {'input': [], 'output': [], 'brake': [], 'waypoints': [], 'target_point': [], 'light': []}

        if self.config['training']['input_ego']:
            data = label_raw
        else:
            data = label_raw[1:] # remove first element (ego vehicle)

        data_car = [[
            1., # type indicator for cars
            float(x['position'][0])-float(label_raw[0]['position'][0]),
            float(x['position'][1])-float(label_raw[0]['position'][1]),
            float(x['yaw'] * 180 / 3.14159265359), # in degrees
            float(x['speed'] * 3.6), # in km/h
            float(x['extent'][2]),
            float(x['extent'][1]),
            ] for x in data if x['class'] == 'Car']
        # if we use the far_node as target waypoint we need the route as input
        data_route = [
            [
                2., # type indicator for route
                float(x['position'][0])-float(label_raw[0]['position'][0]),
                float(x['position'][1])-float(label_raw[0]['position'][1]),
                float(x['yaw'] * 180 / 3.14159265359),  # in degrees
                float(x['id']),
                float(x['extent'][2]),
                float(x['extent'][1]),
            ] 
            for j, x in enumerate(data)
            if x['class'] == 'Route' 
            and float(x['id']) < self.config['training']['max_NextRouteBBs']]
        
        # we split route segment slonger than 10m into multiple segments
        # improves generalization
        data_route_split = []
        for route in data_route:
            if route[6] > 10:
                routes = split_large_BB(route, len(data_route_split))
                data_route_split.extend(routes)
            else:
                data_route_split.append(route)

        data_route = data_route_split[:self.config['training']['max_NextRouteBBs']]

        assert len(data_route) <= self.config['training']['max_NextRouteBBs'], 'Too many routes'

        if self.config['training']['remove_velocity'] == 'input':
            for i in range(len(data_car)):
                data_car[i][4] = 0.

        if self.config['training']['route_only_wp']:
            for i in range(len(data_route)):
                data_route[i][3] = 0.
                data_route[i][-2] = 0.
                data_route[i][-1] = 0.

        features = data_car + data_route

        sample['input'] = features

        # dummy data
        sample['output'] = features
        sample['light'] = self.traffic_light_hazard

        local_command_point = np.array([input_data['target_point'][0], input_data['target_point'][1]])
        sample['target_point'] = local_command_point

        batch = [sample]
        
        input_batch = generate_batch(batch)
        
        self.data = data
        self.data_car = data_car
        self.data_route = data_route
        
        return input_batch

    def destroy(self):
        super().destroy()
        del self.net


def create_BEV(labels_org, gt_traffic_light_hazard, target_point, pred_wp, pix_per_m=5):

    pred_wp = np.array(pred_wp.squeeze())
    s=0
    max_d = 30
    size = int(max_d*pix_per_m*2)
    origin = (size//2, size//2)
    PIXELS_PER_METER = pix_per_m

    # color = [(255, 0, 0), (0, 0, 255)]
    color = [(255), (255)]

    # create black image
    image_0 = Image.new('L', (size, size))
    image_1 = Image.new('L', (size, size))
    image_2 = Image.new('L', (size, size))
    vel_array = np.zeros((size, size))
    draw0 = ImageDraw.Draw(image_0)
    draw1 = ImageDraw.Draw(image_1)
    draw2 = ImageDraw.Draw(image_2)

    draws = [draw0, draw1, draw2]
    imgs = [image_0, image_1, image_2]
    
    for ix, sequence in enumerate([labels_org]):
               
        # features = rearrange(features, '(vehicle features) -> vehicle features', features=4)
        for ixx, vehicle in enumerate(sequence):
            # draw vehicle
            # if vehicle['class'] != 'Car':
            #     continue
            
            x = -vehicle['position'][1]*PIXELS_PER_METER + origin[1]
            y = -vehicle['position'][0]*PIXELS_PER_METER + origin[0]
            yaw = vehicle['yaw']* 180 / 3.14159265359
            extent_x = vehicle['extent'][2]*PIXELS_PER_METER/2
            extent_y = vehicle['extent'][1]*PIXELS_PER_METER/2
            origin_v = (x, y)
            
            if vehicle['class'] == 'Car':
                p1, p2, p3, p4 = get_coords_BB(x, y, yaw-90, extent_x, extent_y)
                if ixx == 0:
                    for ix in range(3):
                        draws[ix].polygon((p1, p2, p3, p4), outline=color[0]) #, fill=color[ix])
                    ix = 0
                else:                
                    draws[ix].polygon((p1, p2, p3, p4), outline=color[ix]) #, fill=color[ix])
                
                if 'speed' in vehicle:
                    vel = vehicle['speed']*3 #/3.6 # in m/s # just for visu
                    endx1, endy1, endx2, endy2 = get_coords(x, y, yaw-90, vel)
                    draws[ix].line((endx1, endy1, endx2, endy2), fill=color[ix], width=2)

            elif vehicle['class'] == 'Route':
                ix = 1
                image = np.array(imgs[ix])
                point = (int(x), int(y))
                cv2.circle(image, point, radius=3, color=color[0], thickness=3)
                imgs[ix] = Image.fromarray(image)
                
    for wp in pred_wp:
        x = wp[1]*PIXELS_PER_METER + origin[1]
        y = -wp[0]*PIXELS_PER_METER + origin[0]
        image = np.array(imgs[2])
        point = (int(x), int(y))
        cv2.circle(image, point, radius=2, color=255, thickness=2)
        imgs[2] = Image.fromarray(image)
          
    image = np.array(imgs[0])
    image1 = np.array(imgs[1])
    image2 = np.array(imgs[2])
    x = target_point[0][1]*PIXELS_PER_METER + origin[1]
    y = -(target_point[0][0])*PIXELS_PER_METER + origin[0]  
    point = (int(x), int(y))
    cv2.circle(image, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image1, point, radius=2, color=color[0], thickness=2)
    cv2.circle(image2, point, radius=2, color=color[0], thickness=2)
    imgs[0] = Image.fromarray(image)
    imgs[1] = Image.fromarray(image1)
    imgs[2] = Image.fromarray(image2)
    
    images = [np.asarray(img) for img in imgs]
    image = np.stack([images[0], images[2], images[1]], axis=-1)
    BEV = image

    img_final = Image.fromarray(image.astype(np.uint8))
    if gt_traffic_light_hazard:
        color = 'red'
    else:
        color = 'green'
    img_final = ImageOps.expand(img_final, border=5, fill=color)
    
    ## add rgb image and lidar
    # all_images = np.concatenate((images_lidar, np.array(img_final)), axis=1)
    # all_images = np.concatenate((rgb_image, all_images), axis=0)
    all_images = img_final
    
    Path(f'bev_viz').mkdir(parents=True, exist_ok=True)
    all_images.save(f'bev_viz/{time.time()}_{s}.png')

    # return BEV


def get_coords(x, y, angle, vel):
    length = vel
    endx2 = x + length * math.cos(math.radians(angle))
    endy2 = y + length * math.sin(math.radians(angle))

    return x, y, endx2, endy2  


def get_coords_BB(x, y, angle, extent_x, extent_y):
    endx1 = x - extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy1 = y + extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx2 = x + extent_x * math.sin(math.radians(angle)) - extent_y * math.cos(math.radians(angle))
    endy2 = y - extent_x * math.cos(math.radians(angle)) - extent_y * math.sin(math.radians(angle))

    endx3 = x + extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy3 = y - extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    endx4 = x - extent_x * math.sin(math.radians(angle)) + extent_y * math.cos(math.radians(angle))
    endy4 = y + extent_x * math.cos(math.radians(angle)) + extent_y * math.sin(math.radians(angle))

    return (endx1, endy1), (endx2, endy2), (endx3, endy3), (endx4, endy4)