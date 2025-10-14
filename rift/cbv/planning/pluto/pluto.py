#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : pluto.py
@Date    : 2024/10/24
'''
from collections import defaultdict
import torch
import numpy as np
import numpy.typing as npt

from typing import Any, Dict, List, Tuple
from scipy.special import softmax

from rift.cbv.planning.base_policy import CBVBasePolicy
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.cbv.planning.pluto.model.pluto_model import PlanningModel
from rift.cbv.planning.pluto.controller.pid_controller import PIDController
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.util.logger import Logger
from rift.util.torch_util import CUDA, get_device_name


class PLUTO(CBVBasePolicy):
    name = 'pluto'
    type = 'il'

    def __init__(self, config, logger: Logger):
        super().__init__(config, logger)
        self.logger = logger
        self.radius = config['obs']['radius']
        self.num_scenario = config['num_scenario']

        self._use_prediction = config['use_prediction']
        self._topk = config['topk']
        self._ckpt_path = config['ckpt_path']
        
        self._frame_rate = config['frame_rate']  # default 10 fps
        self._step_interval = 1.0 / self._frame_rate  # default 0.1s
        self.device = get_device_name()
        
        # the pluto model for inference
        self.pluto_model = CUDA(PlanningModel(radius=self.radius))

        # the PID controller dict (each CBV need a unique PID controller)
        self.controllers = defaultdict(lambda: defaultdict(lambda: PIDController(sample_interval=self._frame_rate)))

        # render
        self._render = config['need_video_render']
        if self._render:
            self.reset_render_data()

    def reset_render_data(self):
        self._render_data = {env_id: {
            "ego_states": {},
            "nearby_agents_states": {},
            "CBV_states": {},
            "route_ids_list": [],
            "reference_lines_list": [],
            "route_waypoints_list": [],
            "interaction_wp_list": [],
            "planning_trajectory_list": [],
            "candidate_trajectories_list": [],
            "candidate_index_list": [],
            "predictions_list": [],
        } for env_id in range(self.num_scenario)}

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            raise ValueError(f'Pluto policy not support training mode.')
        elif mode == 'eval':
            self.pluto_model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, CBVs_obs_list, infos, deterministic=False) -> Dict[str, List[Dict[Any, Any]]]:
        CBVs_action = [{} for _ in range(self.num_scenario)]

        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            if not CBVs_obs:
                continue
            
            env_id = info['env_id']
            pluto_feature_data = PlutoFeature.collate([CBV_obs['raw_pluto_feature'] for CBV_obs in CBVs_obs.values()]).to_device(self.device).data
            with torch.no_grad():
                pluto_output = self.pluto_model(pluto_feature_data)
            for index, (CBV_id, CBV_obs) in enumerate(CBVs_obs.items()):
                CBVs_action[env_id][CBV_id] = self._get_action(pluto_output, CBV_obs, env_id, CBV_id, index)

        # render part
        if self._render:
            for info in infos:
                env_id = info['env_id']
                ego = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id)
                ego_state = CarlaDataProvider.get_current_state(ego)
                nearby_agents = CarlaDataProvider.get_ego_nearby_agents(ego.id)
                nearby_agents_states = {agent.id: CarlaDataProvider.get_current_state(agent) for agent in nearby_agents}

                self._render_data[env_id].update({
                    'ego_states': {ego.id: ego_state},
                    'nearby_agents_states': nearby_agents_states
                })   
        
        # clean useless CBVs
        self._clean_CBVs(infos, CBVs_obs_list)

        data = {
            'CBVs_actions': CBVs_action,
        }
        return data
    
    def _clean_CBVs(self, infos, CBVs_obs_list):
        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            env_id = info['env_id']
            active_ids = set(CBVs_obs.keys())
            if env_id in self.controllers:
                # remove unused CBVs and keep default controller
                for cbv_id in list(self.controllers[env_id].keys()):
                    if cbv_id not in active_ids:
                        del self.controllers[env_id][cbv_id]
                
                if not self.controllers[env_id]:
                    del self.controllers[env_id]

    def get_render_data(self, env_id):
        return self._render_data[env_id]

    def load_infer_checkpoint(self, checkpoint: str, device_name: str):
        ckpt = torch.load(checkpoint, map_location=device_name)
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        return state_dict
    
    def load_train_checkpoint(self, checkpoint: str, device_name: str):
        ckpt = torch.load(checkpoint, map_location=device_name)
        return ckpt['state_dict']

    def load_model(self, resume=True):
        # load the model
        self.pluto_model.load_state_dict(self.load_infer_checkpoint(self._ckpt_path, device_name=self.device))
    
    def save_model(self, episode):
        raise NotImplementedError()
    
    def _get_action(self, pluto_output, CBV_obs, env_id, CBV_id, index):
        CBV = CarlaDataProvider.get_actor_by_id(CBV_id)
        CBV_history_states = CarlaDataProvider.get_history_state(CBV)
        CBV_state = CBV_history_states[-1]

        candidate_trajectories = (
            pluto_output["candidate_trajectories"][index].cpu().numpy().astype(np.float64)
        )
        probability = pluto_output["probability"][index].cpu().numpy()

        if self._use_prediction:
            predictions = pluto_output["output_prediction"][index].cpu().numpy()
        else:
            predictions = None

        ref_free_trajectory = (
            (pluto_output["output_ref_free_trajectory"][index].cpu().numpy().astype(np.float64))
            if "output_ref_free_trajectory" in pluto_output
            else None
        )

        candidate_trajectories, learning_based_score, _, _, _ = self._trim_candidates(
            candidate_trajectories,
            probability,
            CBV_state,
            ref_free_trajectory,
        )

        # select the best trajectory only based on learing based score
        best_candidate_idx = learning_based_score.argmax()

        trajectory = candidate_trajectories[best_candidate_idx, 1:]  # [79, 3]
        local_trajectory = self._global_to_local(trajectory, CBV_state)

        # get the control
        throttle, steer, brake = self.get_control(env_id=env_id, CBV_id=CBV_id, center_state=CBV_state, local_trajectory=local_trajectory)

        if self._render:
            self._render_data[env_id]["CBV_states"][CBV_id] = CBV_state
            self._render_data[env_id]["route_ids_list"].append(CBV_obs['route_ids'])
            self._render_data[env_id]["reference_lines_list"].append(CBV_obs['reference_lines'])
            self._render_data[env_id]["route_waypoints_list"].append(CBV_obs['route_waypoints'])
            self._render_data[env_id]["interaction_wp_list"].append(CBV_obs['interaction_wp'])
            self._render_data[env_id]["planning_trajectory_list"].append(trajectory)
            self._render_data[env_id]["candidate_trajectories_list"].append(candidate_trajectories)
            self._render_data[env_id]["candidate_index_list"].append(best_candidate_idx)
            self._render_data[env_id]["predictions_list"].append(predictions)

        return throttle, steer, brake
    
    def _trim_candidates(
        self,
        candidate_trajectories: np.ndarray,
        probability: np.ndarray,
        ego_state: CarlaAgentState,
        ref_free_trajectory: np.ndarray = None,
    ) -> npt.NDArray[np.float32]:
        """
        Args:
            candidate_trajectories: (n_ref, n_mode, 80, 3)
            probability: (n_ref, n_mode)
        Return:
            sorted_candidate_trajectories: (n_ref * n_mode, 80, 3)
            sorted_probability: (n_ref * n_mode)
        """

        n_ref, n_mode, T, C = candidate_trajectories.shape
        candidate_trajectories = candidate_trajectories.reshape(-1, T, C)
        flatten_probability = probability.reshape(-1)
        orig_indices = np.arange(n_ref * n_mode)

        sorted_idx = np.argsort(-flatten_probability)
        sorted_candidate_trajectories = candidate_trajectories[sorted_idx][: self._topk]
        sorted_probability = flatten_probability[sorted_idx][: self._topk]
        sorted_probability = softmax(sorted_probability)
        sorted_orig_indices = orig_indices[sorted_idx][: self._topk]

        if ref_free_trajectory is not None:
            sorted_candidate_trajectories = np.concatenate(
                [sorted_candidate_trajectories, ref_free_trajectory[None, ...]],
                axis=0,
            )
            sorted_probability = np.concatenate([sorted_probability, [0.25]], axis=0)
            sorted_orig_indices = np.concatenate([sorted_orig_indices, [-1]], axis=0)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        sorted_candidate_trajectories[..., :2] = (
            np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        sorted_candidate_trajectories = np.concatenate(
            [sorted_candidate_trajectories[..., 0:1, :], sorted_candidate_trajectories],
            axis=-2,
        )

        return sorted_candidate_trajectories, sorted_probability, sorted_orig_indices, n_ref, n_mode

    def get_control(self, env_id, CBV_id, center_state: CarlaAgentState, local_trajectory: np.ndarray) -> Tuple[float, float, float]:
        local_pos = local_trajectory[:, :2]  # only take the position info

        speed = center_state.dynamic_car_state.center_velocity_2d.magnitude()  # center forward speed

        controller = self.controllers[env_id][CBV_id]
        
        # get the PID control
        throttle, steer, brake = controller.control_pid(local_pos, speed)

        return throttle, steer, brake
    
    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: CarlaAgentState):

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        # force strict alignment
        delta = origin - global_trajectory[0, :2]
        adjusted_global_pos = global_trajectory[..., :2] + delta

        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(adjusted_global_pos - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)