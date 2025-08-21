#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : pluto_feature_builder.py
@Date    : 2024/10/18
'''
import warnings
from typing import List, Set

import carla
import numpy as np
from rift.cbv.planning.pluto.utils.cost_map_manager import CostMapManager
from nuplan_plugin.actor_state.state_representation import Point2D
from nuplan_plugin.actor_state.tracked_objects_types import TrackedObjectType
from nuplan_plugin.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusType,
)

from shapely import LineString, Point

from rift.cbv.planning.pluto.utils.nuplan_map_utils import CarlaMap
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.cbv.planning.route_planner.route_planner import CBVRoutePlanner
from .pluto_feature import PlutoFeature
from .common import rotate_round_z_axis


class PlutoFeatureBuilder():
    def __init__(self, config, route_planner) -> None:

        self.config = config
        self.obs_config = self.config['obs']
        self.max_agent = self.obs_config['max_agent']
        self.radius = self.obs_config['radius']
        self.history_horizon = self.obs_config['history_horizon']
        self.frame_rate = CarlaDataProvider.get_frame_rate()
        self.history_samples = int(self.history_horizon * self.frame_rate)

        self.map_api: CarlaMap = CarlaDataProvider.get_map_api()
        self.sample_points = self.map_api.map_sample_points
        self.lane_speed_limit_mps = self.map_api.speed_limit_mps

        self.route_planner: CBVRoutePlanner = route_planner

        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]
        self.static_objects_types = [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        self.polygon_types = [
            SemanticMapLayer.LANE,
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.CROSSWALK,
        ]

    def build_feature(self, center, center_nearby_agents, mode:str =None):
        data = {}
        history_horizon_samples = self.history_samples + 1
        history_states: List[CarlaAgentState] = list(CarlaDataProvider.get_history_state(center))[-history_horizon_samples:]
        current_state = history_states[-1]
        query_xy = current_state.center

        data["current_state"] = self.process_current_agent_state(current_state)

        center_features = self.get_center_agent_features(history_states, center)
        center_agent_features, center_agent_tokens, center_agent_polygons = self.get_agent_features(
            query_xy=query_xy,
            center_nearby_agents=center_nearby_agents,
            history_horizon_samples=history_horizon_samples
        )

        data["agent"] = {}
        for k in center_agent_features.keys():
            data["agent"][k] = np.concatenate(
                [center_features[k][None, ...], center_agent_features[k]], axis=0
            )
        center_agent_tokens = ["ego"] + center_agent_tokens
        data["agent_tokens"] = center_agent_tokens

        data["static_objects"] = self.get_static_objects_features(current_state)

        # update the global waypoints route
        reference_lines, route_elements, route_ids, interaction_wp = self.route_planner.build_reference_line(center, current_state, self.radius)

        data["map"], map_polygon_tokens = self.get_map_features(
            map_api=self.map_api,
            query_xy=Point(current_state.center.x, current_state.center.y),
            road_ids=set(route_ids['road_ids']),
            radius=self.radius,
        )

        data["reference_line"] = self.get_reference_line_features(
            center_features=center_features,
            reference_lines=reference_lines,
        )

        if mode.startswith('train'):
            cost_map_manager = CostMapManager(
                origin=current_state.rear_axle.array,
                angle=current_state.rear_axle.heading,
                height=200,
                width=200,
                resolution=0.2,
                map_api=self.map_api,
            )
            cost_maps = cost_map_manager.build_cost_maps(
                static_objects=[],  # currently don't have any static objects
                agents=center_agent_features,
                agents_polygon=center_agent_polygons,
            )
            data["cost_maps"] = cost_maps["cost_maps"]

        return PlutoFeature.normalize(data, first_time=True, radius=self.radius), route_ids, reference_lines, route_elements, interaction_wp

    @staticmethod
    def process_current_agent_state(current_state: CarlaAgentState):
        """
        Args:
            current_state: the current state of the agent
        """
        state = np.zeros(7, dtype=np.float64)
        state[0:2] = current_state.rear_axle.array
        state[2] = current_state.rear_axle.heading
        state[3] = current_state.dynamic_car_state.rear_axle_velocity_2d.x
        state[4] = current_state.dynamic_car_state.rear_axle_acceleration_2d.x
        state[5] = current_state.tire_steering_angle
        state[6] = current_state.dynamic_car_state.angular_velocity

        return state

    def get_center_agent_features(self, history_states: List[CarlaAgentState], agent: carla.Vehicle):
        """
        note that rear axle velocity and acceleration are in center vehicle local frame,
        and need to be transformed to the global frame.
        """
        T = len(history_states)

        position = np.zeros((T, 2), dtype=np.float64)
        heading = np.zeros(T, dtype=np.float64)
        velocity = np.zeros((T, 2), dtype=np.float64)
        acceleration = np.zeros((T, 2), dtype=np.float64)
        shape = np.zeros((T, 2), dtype=np.float64)
        valid_mask = np.ones(T, dtype=np.bool_)

        for t, state in enumerate(history_states):
            position[t] = state.rear_axle.array
            heading[t] = state.rear_axle.heading
            velocity[t] = rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_velocity_2d.array,
                -state.rear_axle.heading,
            )  # world to local
            acceleration[t] = rotate_round_z_axis(
                state.dynamic_car_state.rear_axle_acceleration_2d.array,
                -state.rear_axle.heading,
            )  # world to local
            agent_extent = agent.bounding_box.extent
            shape[t] = np.array([agent_extent.y * 2., agent_extent.x * 2.])  # width and length

        category = np.array(
            self.interested_objects_types.index(TrackedObjectType.EGO), dtype=np.int8
        )

        return {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "acceleration": acceleration,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

    def get_agent_features(
            self,
            query_xy: Point2D,
            center_nearby_agents: List[carla.Vehicle],
            history_horizon_samples: int
    ):
        N, T = min(len(center_nearby_agents), self.max_agent), history_horizon_samples

        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=np.bool_)
        polygon = [None] * N

        if N == 0:
            return (
                {
                    "position": position,
                    "heading": heading,
                    "velocity": velocity,
                    "shape": shape,
                    "category": category,
                    "valid_mask": valid_mask,
                },
                [],
                [],
            )

        agent_ids = np.array([agent.id for agent in center_nearby_agents])  # only consider the current nearby agent
        agent_cur_pos = np.array([CarlaDataProvider.get_current_state(agent).center.array for agent in center_nearby_agents])
        distance = np.linalg.norm(agent_cur_pos - query_xy.array[None, :], axis=1)
        agent_ids_sorted = agent_ids[np.argsort(distance)[: self.max_agent]]  # sort the agent ids according to the distance to center agent
        agent_ids_dict = {agent_id: i for i, agent_id in enumerate(agent_ids_sorted)}

        for agent in center_nearby_agents:
            # get specific agent history state
            history_states: List[CarlaAgentState] = list(CarlaDataProvider.get_history_state(agent))[-history_horizon_samples:]
            for t, state in enumerate(history_states):
                # add the specific agent info of specific time state
                idx = agent_ids_dict[agent.id]
                position[idx, t] = state.agent_state.center.array
                heading[idx, t] = state.agent_state.center.heading
                velocity[idx, t] = state.agent_state.velocity.array
                shape[idx, t] = np.array([state.agent_state.box.width, state.agent_state.box.length])
                valid_mask[idx, t] = True

                if t == history_horizon_samples - 1:
                    category[idx] = self.interested_objects_types.index(
                        state.agent_state.tracked_object_type
                    )
                    polygon[idx] = state.agent_state.box.geometry

        agent_features = {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

        return agent_features, list(agent_ids_sorted), polygon

    def get_static_objects_features(self, current_state: CarlaAgentState):
        # currently don't have any static objects
        static_objects = np.zeros((0, 6), dtype=np.float64)
        valid_mask = np.zeros(0, dtype=np.bool_)
        return {
            "position": static_objects[:, :2],
            "heading": static_objects[:, 2],
            "shape": static_objects[:, 3:5],
            "category": static_objects[:, -1],
            "valid_mask": valid_mask,
        }

    def get_map_features(self, map_api: CarlaMap, query_xy: Point, road_ids: Set, radius: float):
        # get all the map object
        map_objects = map_api.query_proximal_map_data(query_xy, radius)
        lane_objects = (
            map_objects[SemanticMapLayer.LANE]
            + map_objects[SemanticMapLayer.LANE_CONNECTOR]
        )
        crosswalk_objects = map_objects[SemanticMapLayer.CROSSWALK]

        object_ids = {int(obj.token_id): idx for idx, obj in enumerate(lane_objects + crosswalk_objects)}
        object_types = (
            [SemanticMapLayer.LANE] * len(map_objects[SemanticMapLayer.LANE])
            + [SemanticMapLayer.LANE_CONNECTOR]
            * len(map_objects[SemanticMapLayer.LANE_CONNECTOR])
            + [SemanticMapLayer.CROSSWALK]
            * len(map_objects[SemanticMapLayer.CROSSWALK])
        )
        # the number of the objects is M (changeable), sample points usually 20 (fixed)
        M, P = len(lane_objects) + len(crosswalk_objects), self.sample_points
        point_position = np.zeros((M, 3, P, 2), dtype=np.float64)
        point_vector = np.zeros((M, 3, P, 2), dtype=np.float64)
        point_side = np.zeros((M, 3), dtype=np.int8)
        point_orientation = np.zeros((M, 3, P), dtype=np.float64)
        polygon_center = np.zeros((M, 3), dtype=np.float64)
        polygon_position = np.zeros((M, 2), dtype=np.float64)
        polygon_orientation = np.zeros(M, dtype=np.float64)
        polygon_type = np.zeros(M, dtype=np.int8)
        polygon_on_route = np.zeros(M, dtype=np.bool_)
        polygon_tl_status = np.zeros(M, dtype=np.int8)
        polygon_speed_limit = np.zeros(M, dtype=np.float64)
        polygon_has_speed_limit = np.zeros(M, dtype=np.bool_)
        polygon_road_block_id = np.zeros(M, dtype=np.int32)

        for lane in lane_objects:
            idx = object_ids[int(lane.token_id)]
            # for each lane, resample the center lane with predefined sample points (20)
            centerline = lane.centerline
            edges = lane.edges

            point_vector[idx] = edges[:, 1:] - edges[:, :-1]
            point_position[idx] = edges[:, :-1]
            point_orientation[idx] = np.arctan2(
                point_vector[idx, :, :, 1], point_vector[idx, :, :, 0]
            )
            point_side[idx] = np.arange(3)

            polygon_center[idx] = np.concatenate(
                [
                    centerline[int(self.sample_points / 2)],
                    [point_orientation[idx, 0, int(self.sample_points / 2)]],
                ],
                axis=-1,
            )
            polygon_position[idx] = centerline[0]
            polygon_orientation[idx] = point_orientation[idx, 0, 0]
            polygon_type[idx] = self.polygon_types.index(object_types[idx])
            polygon_on_route[idx] = int(lane.road_id) in road_ids  # whether the current map object are in the glabal route's road id set
            polygon_tl_status[idx] = TrafficLightStatusType.GREEN        # assume all the traffic light is Green
            polygon_has_speed_limit[idx] = self.lane_speed_limit_mps is not None
            polygon_speed_limit[idx] = self.lane_speed_limit_mps
            polygon_road_block_id[idx] = int(lane.road_id)

        for crosswalk in crosswalk_objects:
            idx = object_ids[int(crosswalk.token_id)]
            edges = crosswalk.edges
            point_vector[idx] = edges[:, 1:] - edges[:, :-1]
            point_position[idx] = edges[:, :-1]
            point_orientation[idx] = np.arctan2(
                point_vector[idx, :, :, 1], point_vector[idx, :, :, 0]
            )
            point_side[idx] = np.arange(3)
            polygon_center[idx] = np.concatenate(
                [
                    edges[0, int(self.sample_points / 2)],
                    [point_orientation[idx, 0, int(self.sample_points / 2)]],
                ],
                axis=-1,
            )
            polygon_position[idx] = edges[0, 0]
            polygon_orientation[idx] = point_orientation[idx, 0, 0]
            polygon_type[idx] = self.polygon_types.index(object_types[idx])
            polygon_on_route[idx] = False
            polygon_tl_status[idx] = TrafficLightStatusType.UNKNOWN
            polygon_has_speed_limit[idx] = False

        map_features = {
            "point_position": point_position,
            "point_vector": point_vector,
            "point_orientation": point_orientation,
            "point_side": point_side,
            "polygon_center": polygon_center,
            "polygon_position": polygon_position,
            "polygon_orientation": polygon_orientation,
            "polygon_type": polygon_type,
            "polygon_on_route": polygon_on_route,
            "polygon_tl_status": polygon_tl_status,
            "polygon_has_speed_limit": polygon_has_speed_limit,
            "polygon_speed_limit": polygon_speed_limit,
            "polygon_road_block_id": polygon_road_block_id,
        }
        return map_features, list(object_ids.keys())

    def get_reference_line_features(self, center_features: dict, reference_lines: List):
        n_points = int(self.radius / 1.0)
        position = np.zeros((len(reference_lines), n_points, 2), dtype=np.float64)
        vector = np.zeros((len(reference_lines), n_points, 2), dtype=np.float64)
        orientation = np.zeros((len(reference_lines), n_points), dtype=np.float64)
        valid_mask = np.zeros((len(reference_lines), n_points), dtype=np.bool_)
        future_projection = np.zeros((len(reference_lines), 8, 2), dtype=np.float64)

        center_future = center_features["position"][self.history_samples + 1 :]
        if len(center_future) > 0:
            linestring = [
                LineString(reference_lines[i]) for i in range(len(reference_lines))
            ]
            future_samples = center_future[9::10]  # every 1s
            future_samples = [Point(xy) for xy in future_samples]

        for i, line in enumerate(reference_lines):
            subsample = line[::4][: n_points + 1]
            n_valid = len(subsample)
            position[i, : n_valid - 1] = subsample[:-1, :2]
            vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)
            orientation[i, : n_valid - 1] = subsample[:-1, 2]
            valid_mask[i, : n_valid - 1] = True

            if len(center_future) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for j, future_sample in enumerate(future_samples):
                        future_projection[i, j, 0] = linestring[i].project(
                            future_sample
                        )
                        future_projection[i, j, 1] = linestring[i].distance(
                            future_sample
                        )

        return {
            "position": position,
            "vector": vector,
            "orientation": orientation,
            "valid_mask": valid_mask,
            "future_projection": future_projection,
        }