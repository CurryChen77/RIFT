#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : sft_pluto.py
@Date    : 2024/12/16
'''

import gc
from typing import Any, Dict, List
import numpy as np
import torch

import wandb
from rift.cbv.planning.fine_tuner.rlft.ppo_pluto.ppo_pluto import PPOPlutoModel
from rift.cbv.planning.fine_tuner.sft.sft_pluto import SFTPluto
from rift.cbv.planning.fine_tuner.sft.teacher.teacher_model import TeacherModel
from rift.cbv.planning.fine_tuner.training_builder import TrainingEngine, build_training_engine

from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.util.torch_util import CUDA

class RTRPluto(SFTPluto):
    name = 'rtr_pluto'
    type = 'sft'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        # PPO related
        self.ppo_config = config['ppo']
        self.hidden_dim = self.ppo_config['hidden_dim']
        self.state_dim = self.ppo_config['state_dim']
        self.action_dim = self.ppo_config['action_dim']
        self.clip_epsilon = self.ppo_config['clip_epsilon']
        self.lambda_entropy = self.ppo_config['lambda_entropy']

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            # train model
            self.train_model = CUDA(PPOPlutoModel(
                radius=self.radius, state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim,
                clip_epsilon=self.clip_epsilon, lambda_entropy=self.lambda_entropy)
            )  # PPO fine-tuning pluto need an extra value net
            self.train_model.train()
            # inference model
            self.pluto_model.eval()
            # teacher model (pdm_lite) for training
            self.teacher_model = TeacherModel(self.config, self.logger)
        elif mode == 'eval':
            # only inference model
            self.pluto_model.eval()
            self.teacher_model = None
        else:
            raise ValueError(f'Unknown mode {mode}')

    def get_action(self, CBVs_obs_list, infos, deterministic=False) -> Dict[str, List[Dict[Any, Any]]]:
        CBVs_action = [{} for _ in range(self.num_scenario)]
        CBVs_old_log_prob = [{} for _ in range(self.num_scenario)]
        CBVs_action_mode = [{} for _ in range(self.num_scenario)]

        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            if not CBVs_obs:
                continue
            
            env_id = info['env_id']
            pluto_feature_data = PlutoFeature.collate([CBV_obs['raw_pluto_feature'] for CBV_obs in CBVs_obs.values()]).to_device(self.device).data
            with torch.no_grad():
                pluto_output = self.pluto_model(pluto_feature_data)
            for index, (CBV_id, CBV_obs) in enumerate(CBVs_obs.items()):
                # get the action for the inference
                CBVs_action[env_id][CBV_id], CBVs_old_log_prob[env_id][CBV_id], CBVs_action_mode[env_id][CBV_id] = self._get_action(pluto_feature_data, pluto_output, CBV_obs, env_id, CBV_id, index)

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

        data = {
            'CBVs_actions': CBVs_action,
            'CBVs_actions_old_log_prob': CBVs_old_log_prob,
            'CBVs_actions_mode': CBVs_action_mode
        }
            
        # get the teacher info for the training
        if self.mode == 'train':
            CBVs_teacher_infos = [{} for _ in range(self.num_scenario)]
            CBV_id_set = set()
            for info, CBVs_obs in zip(infos, CBVs_obs_list):
                env_id = info['env_id']
                CBV_id_set.update(CBVs_obs.keys())
                CBVs_teacher_infos[env_id] = {
                    CBV_id: self._get_teacher_infos(CBV_id)
                    for CBV_id in CBVs_obs.keys()
                }
                if self._render:
                    self._render_data[env_id]["CBV_teacher_infos"] = CBVs_teacher_infos[env_id]
            # clean unused teacher model
            self._clean_teacher_model(CBV_id_set)
            data['CBVs_teacher_infos'] = CBVs_teacher_infos

        # clean useless CBVs
        self._clean_CBVs(infos, CBVs_obs_list)

        return data

    def _get_action(self, pluto_feature_data, pluto_output, CBV_obs, env_id, CBV_id, index):
        CBV = CarlaDataProvider.get_actor_by_id(CBV_id)
        CBV_history_states = CarlaDataProvider.get_history_state(CBV)
        CBV_state = CBV_history_states[-1]
        r_valid_mask = pluto_feature_data["reference_line"]["valid_mask"][index].any(-1)  # [valid_R, ]

        candidate_trajectories = (
            pluto_output["candidate_trajectories"][index][r_valid_mask].cpu().numpy().astype(np.float64)
        )  # [valid_R, M, T, 3]
        probability = pluto_output["probability"][index][r_valid_mask].cpu().numpy()  # [valid_R, M]

        if self._use_prediction:
            predictions = pluto_output["output_prediction"][index].cpu().numpy()
        else:
            predictions = None

        ref_free_trajectory = (
            (pluto_output["output_ref_free_trajectory"][index].cpu().numpy().astype(np.float64))
            if "output_ref_free_trajectory" in pluto_output
            else None
        )

        candidate_trajectories, learning_based_score, sorted_orig_indices, n_ref, n_mode = self._trim_candidates(
            candidate_trajectories,
            probability,
            CBV_state,
            ref_free_trajectory,
        )

        # select the best trajectory only based on learing based score
        best_candidate_idx = learning_based_score.argmax()

        # get best candidate old log prob
        old_log_prob = np.log(learning_based_score[best_candidate_idx] + 1e-12)  # take log for the score that have been softmax
        # get best candidate original index
        orig_idx = int(sorted_orig_indices[best_candidate_idx])
        r_index = orig_idx // n_mode
        m_index = orig_idx % n_mode

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

        return (throttle, steer, brake), old_log_prob, (r_index, m_index)

    def train(self, e_i):
        self.logger.log('>> Starting fine-tuning...', color='yellow')
        # rebuild the training engine for each training
        dir_path = self.model_path / self.load_agent_info
        
        # update the training cfg learning rate
        self.cfg.lr = max(self.initial_lr* (self.cfg.cl_lr_decay ** self.current_epoch), self.cfg.min_lr)  # decay the learning rate through close loop training

        # build the training engine
        training_engine: TrainingEngine = build_training_engine(
            cfg=self.cfg,
            dir_path=dir_path,
            carla_episode=e_i,
            torch_module_wrapper=self.train_model,  # copy the training pluto model
            buffer=self.buffer,
        )
        
        # load the current training model ckpt
        training_engine.model.load_state_dict(self.load_train_checkpoint(self.checkpoint, device_name=self.device), strict=False)

        # process_buffer
        training_engine.datamodule.preprocess_buffer(training_engine.model.model)  # RL required

        # starting training
        training_engine.trainer.fit(model=training_engine.model, datamodule=training_engine.datamodule)

        if wandb.run is not None:
            wandb.finish()  # finish the wandb run

        # update the latest ckpt
        self.update_training_ckpt()
        
        # update inference model parameters
        self.pluto_model.load_state_dict(self.load_infer_checkpoint(self.checkpoint, device_name=self.device))

        # clean the GPU memory
        del training_engine
        gc.collect()
        torch.cuda.empty_cache()
        # reset buffer
        self.buffer.reset_buffer()
        self.logger.log('>> Finishing fine-tuning...', color='yellow')

    def load_infer_checkpoint(self, checkpoint: str, device_name: str):
        ckpt = torch.load(checkpoint, map_location=device_name)
        state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
        
        # inference model don't need the value net
        value_net_keys = [k for k in state_dict.keys() if k.startswith('value_net')]
        if value_net_keys:
            for k in value_net_keys:
                del state_dict[k]
        
        return state_dict