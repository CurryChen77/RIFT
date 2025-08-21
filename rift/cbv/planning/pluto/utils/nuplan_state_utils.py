#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : nuplan_utils.py
@Date    : 2024/09/27
"""
import math
import carla
import numba
import numpy as np
from typing import List, Optional
from nuplan_plugin.actor_state.oriented_box import OrientedBox
from nuplan_plugin.actor_state.scene_object import SceneObjectMetadata
from nuplan_plugin.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan_plugin.actor_state.ego_state import EgoState
from nuplan_plugin.actor_state.agent_state import AgentState
from nuplan_plugin.actor_state.car_footprint import CarFootprint
from nuplan_plugin.actor_state.dynamic_car_state import DynamicCarState, get_acceleration_shifted, get_velocity_shifted
from nuplan_plugin.actor_state.tracked_objects_types import TrackedObjectType
from nuplan_plugin.actor_state.vehicle_parameters import VehicleParameters
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class CarlaAgentState(EgoState):
    """
        Represent the current state of carla agent, along with its dynamic attributes.
    """

    def __init__(
            self,
            car_footprint: CarFootprint,
            dynamic_car_state: DynamicCarState,
            agent_state: AgentState,
            tire_steering_angle: float,
            is_in_auto_mode: bool,
            time_point: TimePoint
    ):
        super().__init__(car_footprint, dynamic_car_state, tire_steering_angle, is_in_auto_mode, time_point)
        self.agent_state = agent_state

    @classmethod
    def build_from_agent(cls, center_vehicle: carla.Vehicle, vehicle_parameters: VehicleParameters, vehicle_type: TrackedObjectType, time_point: TimePoint):
        center_trans = CarlaDataProvider.get_transform(center_vehicle)

        center_velocity = CarlaDataProvider.get_velocity(center_vehicle)

        center_acc = CarlaDataProvider.get_acceleration(center_vehicle)

        angular_vel = -np.deg2rad(center_vehicle.get_angular_velocity().z)  # from degree/s to rad/s

        # necessary value of Ego state
        center = StateSE2(x=center_trans.location.x, y=-center_trans.location.y, heading=-np.deg2rad(center_trans.rotation.yaw))
        center_velocity_2d = StateVector2D(x=center_velocity.x, y=-center_velocity.y)  # y is negative due to the left-hand coordinate system in CARLA
        center_acceleration_2d = StateVector2D(x=center_acc.x, y=-center_acc.y)  # y is negative due to the left-hand coordinate system in CARLA

        angular_accel = 0.0  # no API to get the angular acceleration, assumed to be 0

        # Build the car footprint from the center
        car_footprint = CarFootprint.build_from_center(center, vehicle_parameters)

        # Calculate the displacement to the rear axle
        rear_axle_to_center_dist = car_footprint.rear_axle_to_center_dist
        displacement = StateVector2D(-rear_axle_to_center_dist, 0.0)

        # Shift velocities and accelerations to the rear axle
        rear_axle_velocity_2d = get_velocity_shifted(displacement, center_velocity_2d, angular_vel)
        rear_axle_acceleration_2d = get_acceleration_shifted(
            displacement, center_acceleration_2d, angular_vel, angular_accel
        )

        # Build the dynamic car state
        dynamic_car_state = DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=rear_axle_to_center_dist,    # global
            rear_axle_velocity_2d=rear_axle_velocity_2d,          # global
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,  # global
            angular_velocity=angular_vel,
            angular_acceleration=angular_accel,
        )

        # agent state for normal background vehicle
        agent_extent = center_vehicle.bounding_box.extent
        agent_state = AgentState(
            metadata=SceneObjectMetadata(token=center_vehicle.type_id, track_token=center_vehicle.type_id, track_id=-1, timestamp_us=time_point.time_us),
            tracked_object_type=vehicle_type,
            oriented_box=OrientedBox(center, agent_extent.x * 2., agent_extent.y * 2., agent_extent.z * 2.),
            velocity=center_velocity_2d,
        )

        return cls(
            car_footprint=car_footprint,
            dynamic_car_state=dynamic_car_state,
            tire_steering_angle=0.0,  # assume to be 0.0
            time_point=time_point,
            is_in_auto_mode=True,
            agent_state=agent_state
        )


@numba.njit
def rotate_round_z_axis(points: np.ndarray, angle: float):
    rotate_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        # dtype=np.float64,
    )
    # return np.matmul(points, rotate_mat)
    return points @ rotate_mat


def get_sample_ego_state(center: Optional[StateSE2] = None, time_us: Optional[int] = 0) -> EgoState:
    """
    Creates a sample EgoState.
    :param center: Vehicle's position. If none it uses the same position returned by get_sample_pose()
    :param time_us: Time in microseconds
    :return: A sample EgoState with arbitrary parameters
    """
    return EgoState(
        car_footprint=get_sample_car_footprint(center),
        dynamic_car_state=get_sample_dynamic_car_state(),
        tire_steering_angle=0.2,
        time_point=TimePoint(time_us),
        is_in_auto_mode=False,
    )


def get_sample_car_footprint(center: Optional[StateSE2] = None) -> CarFootprint:
    """
    Creates a sample CarFootprint.
    :param center: Vehicle's position. If none it uses the same position returned by get_sample_pose()
    :return: A sample CarFootprint with arbitrary parameters
    """
    if center:
        return CarFootprint.build_from_center(center=center, vehicle_parameters=get_pacifica_parameters())
    else:
        return CarFootprint.build_from_center(
            center=get_sample_oriented_box().center, vehicle_parameters=get_pacifica_parameters()
        )


def get_sample_dynamic_car_state(rear_axle_to_center_dist: float = 1.44) -> DynamicCarState:
    """
    Creates a sample DynamicCarState.
    :param rear_axle_to_center_dist: distance between rear axle and center [m]
    :return: A sample DynamicCarState with arbitrary parameters
    """
    return DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist, StateVector2D(1.0, 2.0), StateVector2D(0.1, 0.2)
    )


def get_sample_pose() -> StateSE2:
    """
    Creates a sample SE2 Pose.
    :return: A sample SE2 Pose with arbitrary parameters
    """
    return StateSE2(1.0, 2.0, math.pi / 2.0)


def get_sample_oriented_box() -> OrientedBox:
    """
    Creates a sample OrientedBox.
    :return: A sample OrientedBox with arbitrary parameters
    """
    return OrientedBox(get_sample_pose(), 4.0, 2.0, 1.5)


def get_pacifica_parameters() -> VehicleParameters:
    """
    :return VehicleParameters containing parameters of Pacifica Vehicle.
    """
    return VehicleParameters(
        vehicle_name="pacifica",
        vehicle_type="gen1",
        width=1.1485 * 2.0,
        front_length=4.049,
        rear_length=1.127,
        wheel_base=3.089,
        cog_position_from_rear_axle=1.67,
        height=1.777,
    )
