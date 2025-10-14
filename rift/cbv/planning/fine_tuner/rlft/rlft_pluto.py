#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : rlft_pluto.py
@Date    : 2025/1/24
'''
import copy
import gc
import re
import wandb
import hydra
import numpy as np
import torch
import hydra._internal.instantiate._instantiate2
import hydra.types

from pathlib import Path
from typing import Any, Dict, List
from rift.cbv.planning.fine_tuner.training_builder import TrainingEngine, build_training_engine
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.cbv.planning.pluto.model.pluto_model import PlanningModel
from rift.cbv.planning.pluto.pluto import PLUTO
from rift.cbv.planning.route_planner.route_planner import CBVRoutePlanner
from rift.gym_carla.buffer.cbv_rollout_buffer import CBVRolloutBuffer
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.util.torch_util import CUDA

# Instantiation related symbols
instantiate = hydra._internal.instantiate._instantiate2.instantiate


class RLFTPluto(PLUTO):
    name = 'rlft_pluto'
    type = 'rlft'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.config_path = config['rlft_config_path']
        self.config_name = config['rlft_config_name']
        self.model_path = Path(config['ROOT_DIR']) / config['model_path']
        self.cbv_recog = config['cbv_recog']
        self.seed = config['seed']
        self.pretrain_seed = config['pretrain_seed']
        self.checkpoint = None

        hydra.initialize(config_path=self.config_path, version_base=None)
        self.cfg = hydra.compose(
            config_name=self.config_name,
            overrides=[
                f"+output_dir={self.logger.output_dir}",
                f"frame_rate={self._frame_rate}",
                f"+name={self.logger.output_dir.name}",
            ]
        )
        self.initial_lr = self.cfg.lr  # initial learning rate

        if config['mode'] == 'train_cbv':
            self.load_agent_info = config['ego_policy'] + '-' + str(self.cbv_recog) + '-seed' + str(self.seed)
        else:
            self.load_agent_info = config['pretrain_ego'] + '-' + str(self.cbv_recog) + '-seed' + str(self.pretrain_seed)

        self.save_agent_info = config['ego_policy'] + '-' + str(self.cbv_recog) + '-seed' + str(self.seed)

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
            # build the train model
            self.train_model = CUDA(PlanningModel(radius=self.radius))
            self.train_model.train()
            # inference model
            self.pluto_model.eval()
        elif mode == 'eval':
            self.pluto_model.eval()
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
                (
                    CBVs_action[env_id][CBV_id],
                    CBVs_old_log_prob[env_id][CBV_id],
                    CBVs_action_mode[env_id][CBV_id]
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
            'CBVs_actions_old_log_prob': CBVs_old_log_prob,
            'CBVs_actions_mode': CBVs_action_mode
        }

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

    def set_route_planner(self, route_planner:CBVRoutePlanner):
        self.route_planner = route_planner

    def get_render_data(self, env_id):
        return self._render_data[env_id]

    def set_buffer(self, buffer, total_routes):
        self.buffer: CBVRolloutBuffer = buffer

    def train(self, e_i):
        self.logger.log('>> Starting fine-tuning...', color='yellow')
        # rebuild the training engine for each training
        dir_path = self.model_path / self.load_agent_info

        # update the training cfg learning rate
        self.cfg.lr = max(self.initial_lr * (self.cfg.cl_lr_decay ** self.current_epoch), self.cfg.min_lr)  # decay the learning rate through close loop training

        # build the training engine
        training_engine: TrainingEngine = build_training_engine(
            cfg=self.cfg,
            dir_path=dir_path,
            carla_episode=e_i,
            torch_module_wrapper=self.train_model,  # copy the training pluto model
            buffer=self.buffer,
        )
        
        # load the current training model ckpt
        training_engine.model.load_state_dict(self.load_train_checkpoint(self.checkpoint, device_name=self.device))

        # process_buffer
        training_engine.datamodule.preprocess_buffer()

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

    def update_training_ckpt(self):
        load_dir = self.model_path / self.load_agent_info
        try:
            pattern = re.compile(r"carla_episode=(\d+)")
            episode_files = sorted(
                load_dir.glob("*.ckpt"),
                key=lambda file: int(pattern.search(file.stem).group(1)),
                reverse=True
            )
            self.current_epoch = len(episode_files)
            if episode_files:
                self.checkpoint = episode_files[0].as_posix()
                self.logger.log(f"Updated checkpoint: {episode_files[0].name}", 'green')
            else:
                self.logger.log("No checkpoint files found.", 'yellow')

        except (AttributeError, ValueError, Exception) as e:
            self.logger.log(f"Error processing checkpoint files: {e}", 'red')
            raise

    def load_model(self, resume=True):
        load_dir = self.model_path / self.load_agent_info
        episode_files = list(load_dir.glob(f"*.ckpt"))

        if resume and episode_files:
            latest_file = max(
                episode_files,
                key=lambda file: int(re.search(r"carla_episode=(\d+)", file.stem).group(1))
            )
            self.checkpoint = latest_file.as_posix()  # use the latest model
            self.continue_episode = int(re.search(r"carla_episode=(\d+)", latest_file.stem).group(1))
            self.current_epoch = len(episode_files)  # update the current epoch
            self.logger.log(f">> Loading {self.name} model from {latest_file.name}", 'yellow')
        else:
            if not resume:
                self.logger.log(f">> Clearning all the data in {load_dir}", 'red')
                for file in episode_files:
                    file.unlink()  # remove all the model files
            self.checkpoint = self.config['ckpt_path']  # use the pretrained checkpoint of pluto
            self.logger.log(f">> Using Pluto pretrained model", 'yellow')
            self.continue_episode = 0
            self.current_epoch = 0
        
        # the inference model
        self.pluto_model.load_state_dict(self.load_infer_checkpoint(self.checkpoint, device_name=self.device))

    def save_model(self, episode):
        pass

    def finish(self):
        if wandb.run is not None:
            wandb.finish()