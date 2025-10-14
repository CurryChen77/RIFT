#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : rift_pluto.py
@Date    : 2025/01/27
'''

import numpy as np
import torch

from typing import Any, Dict, List
from rift.cbv.planning.fine_tuner.rlft.traj_eval.traj_evaluator import TrajEvaluator
from rift.cbv.planning.fine_tuner.rlft.rlft_pluto import RLFTPluto
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class RIFTPluto(RLFTPluto):
    name = 'rift_pluto'
    type = 'rlft'

    def __init__(self, config, logger):
        super().__init__(config, logger)

        # multi candidate trajectories evaluator
        self.traj_evaluator = TrajEvaluator(dt=self._step_interval)

    def get_action(self, CBVs_obs_list, infos, deterministic=False) -> Dict[str, List[Dict[Any, Any]]]:
        CBVs_action = [{} for _ in range(self.num_scenario)]
        CBVs_old_group_logits = [{} for _ in range(self.num_scenario)]
        CBVs_group_advantage = [{} for _ in range(self.num_scenario)]

        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            if not CBVs_obs:
                continue
            
            env_id = info['env_id']
            pluto_feature_data = PlutoFeature.collate([CBV_obs['raw_pluto_feature'] for CBV_obs in CBVs_obs.values()]).to_device(self.device).data
            with torch.no_grad():
                pluto_output = self.pluto_model(pluto_feature_data)
            for index, (CBV_id, CBV_obs) in enumerate(CBVs_obs.items()):
                # get the action for the inference
                (
                    CBVs_action[env_id][CBV_id],
                    CBVs_old_group_logits[env_id][CBV_id],
                    CBVs_group_advantage[env_id][CBV_id]
                ) = self._get_action(pluto_feature_data, pluto_output, CBV_obs, env_id, CBV_id, index)
        
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
            'CBVs_actions_old_group_logits': CBVs_old_group_logits,
            'CBVs_group_advantage': CBVs_group_advantage
        }

        return data

    def _get_action(self, pluto_feature_data, pluto_output, CBV_obs, env_id, CBV_id, index):
        CBV = CarlaDataProvider.get_actor_by_id(CBV_id)
        CBV_history_states = CarlaDataProvider.get_history_state(CBV)
        CBV_state = CBV_history_states[-1]

        candidate_trajectories = (
            pluto_output["candidate_trajectories"][index].cpu().numpy().astype(np.float64)
        )
        probability = pluto_output["probability"][index].cpu().numpy()  # [padded_R, M]

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

        # rift require the group advantage
        if self.mode == 'train':
            ego_id = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id).id
            # get group advantage
            nearby_actors = CarlaDataProvider.get_CBV_nearby_agents(ego_id, CBV_id)
            raw_trajectories = pluto_output["trajectory"][index]  # [padded_R, M, T, 6]
            ref_line_pos = pluto_feature_data["reference_line"]["position"][index]  # [padded_R, Ts_r, 2]
            ref_line_angle = pluto_feature_data["reference_line"]["orientation"][index]  # [padded_R, Ts_r]
            ref_line_valid_mask = pluto_feature_data["reference_line"]["valid_mask"][index]  # [padded_R, Ts_r]
            r_valid_mask = ref_line_valid_mask.any(-1)  # [padded_R, ]

            # only pick valid R index
            raw_trajectories = raw_trajectories[r_valid_mask]  # [Valid_R, M, T, 6]
            ref_line_pos = ref_line_pos[r_valid_mask]  # [Valid_R, Ts_r, 2]
            ref_line_angle = ref_line_angle[r_valid_mask]   # [Valid_R, Ts_r]

            # only pick valid time index for ref_line
            ref_line_pos = [pos[mask] for pos, mask in zip(ref_line_pos, ref_line_valid_mask)]
            ref_line_angle = [angle[mask] for angle, mask in zip(ref_line_angle, ref_line_valid_mask)]

            group_advantage = self.traj_evaluator.get_grpo_advantage(
                CBV_history_states,
                raw_trajectories,
                ref_line_pos,
                ref_line_angle,
                nearby_actors
                )  # [Valid_R, M]

            # get old prob (only pick valid R index)
            valid_probability = probability[r_valid_mask.cpu().numpy()]

            old_group_logits = {
                'logits': valid_probability,  # [Valid_R, M]
                'valid_mask': np.ones_like(valid_probability, dtype=np.bool_)  # [Valid_R, M]
            }
        else:
            group_advantage = None
            old_group_logits = None

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

        return (throttle, steer, brake), old_group_logits, group_advantage