#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : data_agent_boxes.py
@Date    : 2023/10/22
"""
import torch
import numpy as np
from rdp import rdp

from rift.ego.expert.autopilot import AutoPilot
import carla

from rift.gym_carla.utils.common import normalize_angle
from rift.gym_carla.utils.utils import get_relative_transform_for_lidar
from rift.gym_carla.visualization.visualize import draw_route


def get_entry_point():
    return 'DataAgent'


class DataAgent(AutoPilot):
    def __init__(self, config=None, logger=None):
        super().__init__(config)

        self.cfg = config

        self.map_precision = 10.0  # meters per point
        self.rdp_epsilon = 0.5  # epsilon for route shortening

        # radius in which other actors/map elements are considered
        # distance is from the center of the ego-vehicle and measured in 3D space
        self.max_actor_distance = self.detection_radius  # copy from expert
        self.max_light_distance = self.light_radius  # copy from expert
        self.max_map_element_distance = 30.0
        
        if self.save_path is not None:
            (self.save_path / 'boxes').mkdir()

    def set_planner(self, ego_vehicle,  global_plan_gps, global_plan_world_coord):
        super().set_planner(ego_vehicle, global_plan_gps, global_plan_world_coord)

    def tick(self, input_data):
        result = super().tick_autopilot(input_data)

        return result

    @torch.no_grad()
    def run_step(self, input_data, sensors=None):

        control = super().run_step(input_data)

        return control

    def destroy(self):
        pass
    
    def lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        transform.location += shift * transform.get_forward_vector()
        return transform

    def get_bev_boxes(self, input_data=None, lidar=None, pos=None, viz_route=None):

        # -----------------------------------------------------------
        # Ego vehicle
        # -----------------------------------------------------------

        # add vehicle velocity and brake flag
        ego_location = self._vehicle.get_location()
        ego_transform = self._vehicle.get_transform()
        ego_control   = self._vehicle.get_control()
        ego_velocity  = self._vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity) # In m/s
        ego_brake = ego_control.brake
        ego_rotation = ego_transform.rotation
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_extent = self._vehicle.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw = ego_rotation.yaw/180*np.pi
        relative_yaw = 0
        relative_pos = get_relative_transform_for_lidar(ego_matrix, ego_matrix)

        results = []

        # add ego-vehicle to results list
        # the format is category, extent*3, position*3, yaw, points_in_bbox, distance, id
        # the position is in lidar coordinates
        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1] ],
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1, 
                  "distance": -1, 
                  "speed": ego_speed, 
                  "brake": ego_brake,
                  "id": int(self._vehicle.id),
                }
        results.append(result)
        
        # -----------------------------------------------------------
        # Other vehicles
        # -----------------------------------------------------------

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        tlights = self._actors.filter('*traffic_light*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(ego_location) < self.max_actor_distance):
                if (vehicle.id != self._vehicle.id):
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())

                    vehicle_extent = vehicle.bounding_box.extent
                    dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
                    yaw =  vehicle_rotation.yaw/180*np.pi

                    relative_yaw = normalize_angle(yaw - ego_yaw)
                    relative_pos = get_relative_transform_for_lidar(ego_matrix, vehicle_matrix)

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control   = vehicle.get_control()
                    vehicle_velocity  = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity) # In m/s
                    vehicle_brake = vehicle_control.brake

                    # filter bbox that didn't contains points of contains less points
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
                        #print("num points in bbox", num_in_bbox_points)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Car",
                        "extent": [dx[2], dx[0], dx[1]],
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points), 
                        "distance": distance, 
                        "speed": vehicle_speed, 
                        "brake": vehicle_brake,
                        "id": int(vehicle.id),
                    }
                    results.append(result)

        # -----------------------------------------------------------
        # Route rdp
        # -----------------------------------------------------------
        if input_data is not None:
            self._waypoint_planner.load()
            waypoint_route = self._waypoint_planner.run_step(pos)
            self.waypoint_route = np.array([[node[0][0],node[0][1]] for node in waypoint_route])

            self._waypoint_planner.save()
        
        
        max_len = 50
        if len(self.waypoint_route) < max_len:
            max_len = len(self.waypoint_route)
        shortened_route = rdp(self.waypoint_route[:max_len], epsilon=self.rdp_epsilon)
        
        # convert points to vectors
        vectors = shortened_route[1:] - shortened_route[:-1]
        midpoints = shortened_route[:-1] + vectors/2.
        norms = np.linalg.norm(vectors, axis=1)
        angles = np.arctan2(vectors[:,1], vectors[:,0])

        for i, midpoint in enumerate(midpoints):
            # find distance to center of waypoint
            center_bounding_box = carla.Location(midpoint[0], midpoint[1], 0.0)
            transform = carla.Transform(center_bounding_box)
            route_matrix = np.array(transform.get_matrix())
            relative_pos = get_relative_transform_for_lidar(ego_matrix, route_matrix)
            distance = np.linalg.norm(relative_pos)
            
            # find distance to beginning of bounding box
            starting_bounding_box = carla.Location(shortened_route[i][0], shortened_route[i][1], 0.0)
            st_transform = carla.Transform(starting_bounding_box)
            st_route_matrix = np.array(st_transform.get_matrix())
            st_relative_pos = get_relative_transform_for_lidar(ego_matrix, st_route_matrix)
            st_distance = np.linalg.norm(st_relative_pos)

            # only store route boxes that are near the ego vehicle
            if i > 0 and st_distance > self.max_route_distance:
                continue

            length_bounding_box = carla.Vector3D(norms[i]/2., ego_extent.y, ego_extent.z)
            bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
            bounding_box.rotation = carla.Rotation(pitch = 0.0,
                                                yaw   = angles[i] * 180 / np.pi,
                                                roll  = 0.0)

            route_extent = bounding_box.extent
            dx = np.array([route_extent.x, route_extent.y, route_extent.z]) * 2.
            relative_yaw = normalize_angle(angles[i] - ego_yaw)

            # visualize subsampled route
            if viz_route:
                draw_route(self._world, bounding_box=bounding_box, frame_rate=self.frame_rate)

            result = {
                "class": "Route",
                "extent": [dx[2], dx[0], dx[1]],
                "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                "yaw": relative_yaw,
                "centre_distance": distance,
                "starting_distance": st_distance,
                "id": i,
            }
            results.append(result)


        # if int(os.environ.get('DATAGEN')):
        #     # -----------------------------------------------------------
        #     # Traffic lights
        #     # -----------------------------------------------------------
        #
        #     _traffic_lights = self.get_nearby_object(ego_location, tlights, self.max_light_distance)
        #
        #     for light in _traffic_lights:
        #         if   (light.state == carla.libcarla.TrafficLightState.Red):
        #             state = 0
        #         elif (light.state == carla.libcarla.TrafficLightState.Yellow):
        #             state = 1
        #         elif (light.state == carla.libcarla.TrafficLightState.Green):
        #             state = 2
        #         else: # unknown
        #             state = -1
        #
        #         center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
        #         center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y, center_bounding_box.z)
        #         length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y, light.trigger_volume.extent.z)
        #         transform = carla.Transform(center_bounding_box) # can only create a bounding box from a transform.location, not from a location
        #         bounding_box = carla.BoundingBox(transform.location, length_bounding_box)
        #
        #         gloabl_rot = light.get_transform().rotation
        #         bounding_box.rotation = carla.Rotation(pitch = light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
        #                                             yaw   = light.trigger_volume.rotation.yaw   + gloabl_rot.yaw,
        #                                             roll  = light.trigger_volume.rotation.roll  + gloabl_rot.roll)
        #
        #         light_rotation = transform.rotation
        #         light_matrix = np.array(transform.get_matrix())
        #
        #         light_extent = bounding_box.extent
        #         dx = np.array([light_extent.x, light_extent.y, light_extent.z]) * 2.
        #         yaw =  light_rotation.yaw/180*np.pi
        #
        #         relative_yaw = normalize_angle(yaw - ego_yaw)
        #         relative_pos = get_relative_transform_for_lidar(ego_matrix, light_matrix)
        #
        #         distance = np.linalg.norm(relative_pos)
        #
        #         result = {
        #             "class": "Light",
        #             "extent": [dx[2], dx[0], dx[1]],
        #             "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
        #             "yaw": relative_yaw,
        #             "distance": distance,
        #             "state": state,
        #             "id": int(light.id),
        #         }
        #         results.append(result)
        #
        #     # -----------------------------------------------------------
        #     # Map elements
        #     # -----------------------------------------------------------
        #
        #     for lane_id, poly in enumerate(self.polygons):
        #         for point_id, point in enumerate(poly):
        #             if (point.location.distance(ego_location) < self.max_map_element_distance):
        #                 point_matrix = np.array(point.get_matrix())
        #
        #                 yaw =  point.rotation.yaw/180*np.pi
        #
        #                 relative_yaw = yaw - ego_yaw
        #                 relative_pos = get_relative_transform_for_lidar(ego_matrix, point_matrix)
        #                 distance = np.linalg.norm(relative_pos)
        #
        #                 result = {
        #                     "class": "Lane",
        #                     "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
        #                     "yaw": relative_yaw,
        #                     "distance": distance,
        #                     "point_id": int(point_id),
        #                     "lane_id": int(lane_id),
        #                 }
        #                 results.append(result)
                    
        return results

    def get_points_in_bbox(self, ego_matrix, vehicle_matrix, dx, lidar):
        # inverse transform lidar to 
        Tr_lidar_2_ego = self.get_lidar_to_vehicle_transform()
        
        # construct transform from lidar to vehicle
        Tr_lidar_2_vehicle = np.linalg.inv(vehicle_matrix) @ ego_matrix @ Tr_lidar_2_ego

        # transform lidar to vehicle coordinate
        lidar_vehicle = Tr_lidar_2_vehicle[:3, :3] @ lidar[1][:, :3].T + Tr_lidar_2_vehicle[:3, 3:]

        # check points in bbox
        x, y, z = dx / 2.
        # why should we use swap?
        x, y = y, x
        num_points = ((lidar_vehicle[0] < x) & (lidar_vehicle[0] > -x) & 
                      (lidar_vehicle[1] < y) & (lidar_vehicle[1] > -y) & 
                      (lidar_vehicle[2] < z) & (lidar_vehicle[2] > -z)).sum()
        return num_points

    def get_lidar_to_vehicle_transform(self):
        # yaw = -90
        rot = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=np.float32)
        T = np.eye(4)

        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.5
        T[:3, :3] = rot
        return T

        
    def get_vehicle_to_lidar_transform(self):
        return np.linalg.inv(self.get_lidar_to_vehicle_transform())

    def get_image_to_vehicle_transform(self):
        # yaw = 0.0 as rot is Identity
        T = np.eye(4)
        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.3

        # rot is from vehicle to image
        rot = np.array([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]], dtype=np.float32)
        
        # so we need a transpose here
        T[:3, :3] = rot.T
        return T

    def get_vehicle_to_image_transform(self):
        return np.linalg.inv(self.get_image_to_vehicle_transform())

    def get_lidar_to_image_transform(self):
        Tr_lidar_to_vehicle = self.get_lidar_to_vehicle_transform()
        Tr_image_to_vehicle = self.get_image_to_vehicle_transform()
        T_lidar_to_image = np.linalg.inv(Tr_image_to_vehicle) @ Tr_lidar_to_vehicle
        return T_lidar_to_image