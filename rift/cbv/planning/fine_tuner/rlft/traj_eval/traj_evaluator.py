#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : forword_simulator.py
@Date    : 2025/02/18
'''

import cv2
import torch
import carla
import numpy as np
import itertools

from typing import List
from shapely import Point, Polygon
from shapely.strtree import STRtree

from rift.cbv.planning.fine_tuner.rlft.traj_eval.track_propogate import TrackPropagate
from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.ego.pdm_lite.config import GlobalConfig
from rift.ego.pdm_lite.kinematic_bicycle_model import KinematicBicycleModel

from rift.gym_carla.reward.reward_model import DenseRewardModel
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from nuplan_plugin.actor_state.state_representation import StateSE2
from nuplan_plugin.maps.maps_datatypes import SemanticMapLayer
from nuplan_plugin.planner.transform_utils import _get_fixed_timesteps, _get_velocity_and_acceleration


DA = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]

def compute_agents_vertices(
    center: np.ndarray,
    angle: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray:
    """
    Args:
        position: (N, T, 2)
        angle: (N, T)
        shape: (N, 2) [width, length]
    Returns:
        4 corners of oriented box (FL, RL, RR, FR)
        vertices: (N, T, 4, 2)
    """
    # Extracting dimensions
    N, T = center.shape[0], center.shape[1]

    # Reshaping the arrays for calculations
    center = center.reshape(N * T, 2)
    angle = angle.reshape(N * T)

    if shape.ndim == 2:
        shape = (shape / 2).repeat(T, axis=0)
    else:
        shape = (shape / 2).reshape(N * T, 2)

    # Calculating half width and half_l
    half_w = shape[:, 0]
    half_l = shape[:, 1]

    # Calculating cos and sin of angles
    cos_angle = np.cos(angle)[:, None]
    sin_angle = np.sin(angle)[:, None]
    rot_mat = np.stack([cos_angle, sin_angle, -sin_angle, cos_angle], axis=-1).reshape(
        N * T, 2, 2
    )

    offset_width = np.stack([half_w, half_w, -half_w, -half_w], axis=-1)
    offset_length = np.stack([half_l, -half_l, -half_l, half_l], axis=-1)

    vertices = np.stack([offset_length, offset_width], axis=-1)
    vertices = np.matmul(vertices, rot_mat) + center[:, None]

    # Calculating vertices
    vertices = vertices.reshape(N, T, 4, 2)

    return vertices


class TrajEvaluator:
    def __init__(
        self,
        dt: float = 0.1,
        num_frames: int = 40,
        sample_interval: int = 5,
        bbox_inflation_ratio = 1.1,
        map_width = 400,
        map_height = 400,
        resolution = 0.5,
        
    ) -> None:
        self.dt = dt
        self.interval = int(dt * 10)
        self.sample_interval = sample_interval
        self.bbox_inflation_ratio = bbox_inflation_ratio
        self.num_frames = num_frames
        self.map_width = map_width
        self.map_height = map_height
        self.resolution = resolution
        self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
        self.offset = np.array([map_height / 2, map_width / 2], dtype=np.float32)

        self.config = GlobalConfig()

        self.center_rollout_model = TrackPropagate(virtual_time_step=self.dt)

        # other vehicle forward simulation
        self.other_vehicle_model = KinematicBicycleModel(self.config)

        # reward model
        self.reward_model = DenseRewardModel()

    def get_center_rollout(self, trajectories: torch.Tensor, center_history_states):
        '''
        args:
            candidate_trajectories: List[InterpolatedTrajectory]
            init_speed: the initial speed of the vehicle
        '''
        # only keey the x, y, heading
        device, dtype = trajectories.device, trajectories.dtype
        heading = torch.atan2(trajectories[..., 3], trajectories[..., 2])  # [R, M, T]
        out_trajectory = torch.cat(
            [trajectories[..., :2], heading[..., np.newaxis]], axis=-1
        )  # [R, M, T, 3]

        # merge the R and M dim
        R, M, T, C = out_trajectory.shape
        out_trajectory = out_trajectory.reshape(-1, T, C)  # [R*M, T, 3]

        # to global
        center_cur_pos = torch.tensor(center_history_states[-1].rear_axle.array, device=device, dtype=dtype)  # [2,]
        center_cur_heading = torch.tensor(center_history_states[-1].rear_axle.heading, device=device, dtype=dtype)

        # force the first point to be (0, 0)
        first_points = out_trajectory[:, 0, :2]
        out_trajectory[:, :, :2] -= first_points.unsqueeze(1)
        
        cos_h = torch.cos(center_cur_heading)
        sin_h = torch.sin(center_cur_heading)
        rot_mat = torch.stack((
            torch.stack([ cos_h,  sin_h], dim=-1),
            torch.stack([-sin_h,  cos_h], dim=-1)
            ), dim=-2
        )  # [2, 2]

        ref_traj_pos = (
            torch.matmul(out_trajectory[..., :2], rot_mat) + center_cur_pos
        )  # [G, Ts, 2]
        ref_traj_heading = out_trajectory[..., 2] + center_cur_heading  # [G, Ts]

        # rollout the reference trajectory
        rollout_center, rollout_angle, rollout_speed, rollout_acc, rollout_angular_vel, rollout_angular_acc, vertices = self.center_rollout_model.propagate(
            ref_traj_pos, ref_traj_heading, center_history_states
        ) 
        
        return rollout_center, rollout_angle, rollout_speed, rollout_acc, rollout_angular_vel, rollout_angular_acc, vertices

    def get_other_vehicle_rollout(self, nearby_actors, num_future_frames=80, near_lane_change=True):
        """
        Predict the future bounding boxes of actors for a given number of frames.

        Args:
            nearby_actors (list): A list of nearby actors.
            num_future_frames (int): The number of future frames to predict.
            near_lane_change (bool): Whether the center vehicle is near a lane change maneuver.
        Returns:
            dict: A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
        """

        # If there are nearby actors, calculate their future bounding boxes
        if nearby_actors:
            # Get the previous control inputs (steering, throttle, brake) for the nearby actors
            previous_controls = [actor.get_control() for actor in nearby_actors]
            previous_actions = np.array(
                [[control.steer, control.throttle, control.brake] for control in previous_controls])

            # Get the current velocities, locations, and headings of the nearby actors
            velocities = np.array([actor.get_velocity().length() for actor in nearby_actors])
            locations = np.array([[actor.get_location().x,
                                    actor.get_location().y,
                                    actor.get_location().z] for actor in nearby_actors])
            headings = np.deg2rad(np.array([actor.get_transform().rotation.yaw for actor in nearby_actors]))

            # Initialize arrays to store future locations, headings, and velocities
            future_locations = np.empty((num_future_frames, len(nearby_actors), 3), dtype="float")
            future_headings = np.empty((num_future_frames, len(nearby_actors)), dtype="float")
            future_velocities = np.empty((num_future_frames, len(nearby_actors)), dtype="float")

            # Forecast the future locations, headings, and velocities for the nearby actors
            for i in range(num_future_frames):
                locations, headings, velocities = self.other_vehicle_model.forecast_other_vehicles(
                    locations, headings, velocities, previous_actions)
                future_locations[i] = locations.copy()
                future_velocities[i] = velocities.copy()
                future_headings[i] = headings.copy()

            # Calculate the predicted bounding boxes for each nearby actor and future frame
            actor_shape = np.empty((num_future_frames, len(nearby_actors), 2), dtype="float")
            for actor_idx, actor in enumerate(nearby_actors):
                for i in range(num_future_frames):
                    # Get the extent (dimensions) of the actor's bounding box
                    extent = actor.bounding_box.extent
                    # Otherwise we would increase the extent of the bounding box of the vehicle
                    extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                    # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                    # uncertainty during forecasting
                    s = self.config.high_speed_min_extent_x_other_vehicle_lane_change if near_lane_change \
                        else self.config.high_speed_min_extent_x_other_vehicle
                    extent.x *= self.config.slow_speed_extent_factor_ego if future_velocities[
                        i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                            s,
                            self.config.high_speed_min_extent_x_other_vehicle * float(i) / float(num_future_frames))
                    extent.y *= self.config.slow_speed_extent_factor_ego if future_velocities[
                        i, actor_idx] < self.config.extent_other_vehicles_bbs_speed_threshold else max(
                            self.config.high_speed_min_extent_y_other_vehicle,
                            self.config.high_speed_extent_y_factor_other_vehicle * float(i) /
                            float(num_future_frames))
                    
                    extent.x *= self.bbox_inflation_ratio
                    extent.y *= self.bbox_inflation_ratio

                    actor_shape[i, actor_idx] = np.array([extent.y*2, extent.x*2]) # [width, length]

            future_locations = future_locations.transpose(1, 0, 2)  # [N, Ts, 3]
            future_headings = future_headings.transpose(1, 0)  # [N, Ts]
            actor_shape = actor_shape.transpose(1, 0, 2)  # [N, Ts, 2]
            
            vertices = compute_agents_vertices(
                center=future_locations[..., :2] * np.array([1, -1]), # left-hand to right-hand
                angle=-future_headings,  # left-hand to right-hand
                shape=actor_shape,
            )  # in global coord, right-hand
        else:
            vertices = np.zeros((0, num_future_frames, 4, 2), dtype=np.float32)

        return vertices

    def get_collision_matrix(self, center_rollout_vertices: np.ndarray, other_vehicle_rollout_vertices: np.ndarray):
        """
        Args:
            center_rollout_vertices: (G, Ts, 4, 2)
            other_vehicle_rollout_vertices: (N, Ts, 4, 2)
        """
        G, Ts, _, _ = center_rollout_vertices.shape
        N, Ts, _, _ = other_vehicle_rollout_vertices.shape

        collision_indices = np.zeros((G, Ts), dtype=np.bool_)
        if N == 0:
            return collision_indices
        
        # for each time step
        for j in range(Ts):
            # Create STRtree for other vehicles at this time step (once per time step)
            other_polygons = [Polygon(other_vehicle_rollout_vertices[n, j]) for n in range(N)]
            tree = STRtree(other_polygons)
            
            # for each center vehicle trajectory, if not already collided
            for i in range(G):
                ego_polygon = Polygon(center_rollout_vertices[i, j])
                
                # Query the STRtree for candidate polygons that intersect
                intersect_indexs = tree.query(ego_polygon)
                
                if intersect_indexs.size > 0:
                    collision_indices[i, j] = True
                    continue

        return collision_indices

    def get_off_road_matrix(self, rollout_center: np.ndarray, center_state: CarlaAgentState):
        """
        Args:
            rollout_center: (G, Ts, 2)
            center_state: CarlaAgentState
        """
        # build drivable area mask
        origin = center_state.center.array
        angle = center_state.center.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )

        off_road_mask = np.ones((self.map_height, self.map_width), dtype=np.uint8)

        radius = max(self.map_height, self.map_width) * self.resolution / 2
        map_api: CarlaMap = CarlaDataProvider.get_map_api()
        objects_dict = map_api.query_proximal_map_data(
            Point(*origin), radius
        )
        da_objects_dict = {da: objects_dict[da] for da in DA}

        da_objects = itertools.chain.from_iterable(da_objects_dict.values())

        for obj in da_objects:
            self.fill_polygon(off_road_mask, obj.polygon, origin, rot_mat, value=0)

        # get off road matrix

        G, T, _ = rollout_center.shape
        all_points = rollout_center.reshape(-1, 2)
        pixel_coords = self.global_to_pixel(all_points, origin, rot_mat)  # (G*T, 2)
    
        pixel_indices = np.round(pixel_coords).astype(int)  # (G*T, 2)
        
        # check if the pixel indices are valid
        valid_x = (pixel_indices[:, 0] >= 0) & (pixel_indices[:, 0] < self.map_width)
        valid_y = (pixel_indices[:, 1] >= 0) & (pixel_indices[:, 1] < self.map_height)
        valid = valid_x & valid_y
        
        # initialize the off road flags
        off_road_flags = np.zeros(all_points.shape[0], dtype=np.bool_)
        
        # check if the pixel is in the offroad
        off_road_flags[valid] = off_road_mask[pixel_indices[valid, 1], pixel_indices[valid, 0]] == 1
        
        off_road_matrix = off_road_flags.reshape(G, T)

        return off_road_matrix
    
    def global_to_pixel(self, coord: np.ndarray, origin, rot_mat):
        coord = np.matmul(coord - origin, rot_mat)
        coord = coord / self.resolution_hw + self.offset
        return coord

    def fill_polygon(self, mask, polygon, origin, rot_mat, value=1):
        polygon = self.global_to_pixel(np.stack(polygon.exterior.coords.xy, axis=1), origin, rot_mat)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], value)

    def get_rollout_return(self, delta_dis, delta_angle, rollout_speed, rollout_acc,
                           rollout_angular_vel, rollout_angular_acc, collision_matrix, off_road_matrix, gamma=0.98):
        """
        Args:
            delta_dis: (G, Ts,)
            delta_angle: (G, Ts,)
            rollout_speed: (G, Ts,)
            rollout_acc: (G, Ts,)
            rollout_angular_vel: (G, Ts,)
            rollout_angular_acc: (G, Ts,)
            collision_matrix: (G, Ts)
            off_road_matrix: (G, Ts)
            gamma: discount factor
        """
        G, Ts = delta_angle.shape

        rollout_return = np.zeros((G,), dtype=np.float64)

        for i in range(G):
            for j in range(Ts):
                collision = collision_matrix[i, j]
                off_road = off_road_matrix[i, j]
                # Calculate the return with discount factor gamma
                rollout_return[i] += self.reward_model.get_reward(
                    abs(delta_dis[i, j]),
                    abs(delta_angle[i, j]),
                    rollout_speed[i, j],
                    rollout_acc[i, j],
                    rollout_angular_vel[i, j],
                    rollout_angular_acc[i, j],
                    int(collision),
                    int(off_road),
                    ) * gamma ** j
                
                if collision:
                    break

        return rollout_return

    def get_ref_line_info(self, trajectories, ref_line_pos, ref_line_angle):
        """
        Args:
            trajectories: tensor [R, M, Ts, C]
            ref_line_pos: R tensor [Ts_r, 2] 
            ref_line_angle: R tensor [Ts_r, 2]
        Return:
            distance_matrix [R*M, Ts]: the closest distance between the trajectory and the reference line
            angle_matrix [R*M, Ts]: the delta angle between the trajectory and the reference line
        """
        R, M, Ts, _ = trajectories.shape
        dtype = trajectories.dtype
        device = trajectories.device

        delta_dis = torch.zeros((R, M, Ts), dtype=dtype, device=device)
        delta_angle = torch.zeros((R, M, Ts), dtype=dtype, device=device)

        for r_idx in range(R):
            ref_pos = ref_line_pos[r_idx]  # [Ts_r, 2] ref line position
            ref_angle = ref_line_angle[r_idx]

            traj = trajectories[r_idx]  # [M, Ts, C] trajectory
            cand_pos = traj[..., :2]  # [M, Ts, 2] pos
            cand_angle = torch.atan2(traj[..., 3], traj[..., 2])  # [M, Ts] angle

            # the distance between the trajectory and the reference line
            diff = cand_pos.unsqueeze(2) - ref_pos.unsqueeze(0).unsqueeze(0)  # [M, Ts, Ts_r, 2]
            dist = torch.norm(diff, dim=-1)  # [M, Ts, Ts_r]

            # find the closest reference point index
            closest_idx = torch.argmin(dist, dim=-1)  # [M, Ts]

            closest_angle = ref_angle[closest_idx]  # [M, Ts]
            angle_diff = cand_angle - closest_angle
            delta_angle[r_idx] = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

            # get the closest distance
            closest_pos = ref_pos[closest_idx]  # [M, Ts, 2]
            rel_pos = cand_pos - closest_pos

            tangent_dir = torch.stack([
                torch.cos(closest_angle), 
                torch.sin(closest_angle)
            ], dim=-1)  # [M, Ts, 2]

            cross_prod = rel_pos[..., 0] * tangent_dir[..., 1] - rel_pos[..., 1] * tangent_dir[..., 0]
            delta_dis[r_idx] = -cross_prod  # [M, Ts]

        return delta_dis.view(-1, Ts).cpu().numpy(), delta_angle.view(-1, Ts).cpu().numpy()

    def get_grpo_advantage(self,
            center_history_states: List[CarlaAgentState],
            trajectories: torch.Tensor,
            ref_line_pos: List[torch.Tensor],
            ref_line_angle: List[torch.Tensor],
            nearby_actors):
        """
        Args:
            center_history_states: history states of the center vehicle
            trajectories: raw trajectories
            ref_line_pos: reference line position
            ref_line_angle: reference line angle
            nearby_actors: nearby actors
        """
        R, M, _, _ = trajectories.shape
        trajectories = trajectories[:, :, :self.num_frames, :]  # [R, M, Ts, C]

        # get delta_angle, delta_dis in local coord
        delta_dis, delta_angle = self.get_ref_line_info(trajectories, ref_line_pos, ref_line_angle)

        # center vehicle forward simulation
        rollout_center, rollout_angle, rollout_speed, rollout_acc, rollout_angular_vel, rollout_angular_acc, rollout_vertices = self.get_center_rollout(trajectories, center_history_states)  # [G, Ts, ...]

        # nearby_agents forward simulation
        nearby_agents_rollout_vertices = self.get_other_vehicle_rollout(nearby_actors, num_future_frames=self.num_frames)  # [N, Ts, ...]

        # check collision between center rollout traj and other vehicles rollout traj
        collision_matrix = self.get_collision_matrix(rollout_vertices, nearby_agents_rollout_vertices)  # [G, Ts]

        # check off road
        off_road_matrix = self.get_off_road_matrix(rollout_center, center_history_states[-1])
        
        # return of each rollout candidate trajectory
        rollout_return = self.get_rollout_return(
            delta_dis,
            delta_angle,
            rollout_speed,
            rollout_acc,
            rollout_angular_vel,
            rollout_angular_acc,
            collision_matrix,
            off_road_matrix
        )  # [G,]

        # GRPO Advantage
        mean_return = np.mean(rollout_return)
        std_return = np.std(rollout_return) + 1e-5
        advantage = (rollout_return - mean_return) / std_return
        advantage = advantage.reshape(R, M)

        return {
            'advantage': advantage,  # [R, M,]
            'valid_mask': np.ones_like(advantage, dtype=np.bool_)  # [R, M,]
        }

