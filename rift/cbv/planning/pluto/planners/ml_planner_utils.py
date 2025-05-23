from copy import deepcopy
from typing import Deque, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from rift.util.torch_util import get_device_name
from nuplan_plugin.actor_state.ego_state import EgoState
from nuplan_plugin.actor_state.state_representation import StateSE2
from nuplan_plugin.planner.transform_utils import (
    _get_fixed_timesteps,
    _get_velocity_and_acceleration,
    _se2_vel_acc_to_ego_state,
)


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def global_trajectory_to_states(
    global_trajectory: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    include_ego_state: bool = False,
):
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    global_states = [StateSE2.deserialize(pose) for pose in global_trajectory]

    velocities, accelerations = _get_velocity_and_acceleration(
        global_states, ego_history, timesteps
    )
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            global_states, velocities, accelerations, timesteps
        )
    ]

    if include_ego_state:
        agent_states.insert(0, ego_state)
    else:
        init_state = deepcopy(agent_states[0])
        init_state._time_point = ego_state.time_point
        agent_states.insert(0, init_state)

    return agent_states

