#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py.py
@Date    : 2024/09/24
"""
from rift.gym_carla.observation.base_observation import EgoBaseObservation, CBVBaseObservation, EgoNormalObservation, CBVNormalObservation, EgoSimpleObservation
from rift.gym_carla.observation.pluto_observation import EgoPlutoObservation, CBVPlutoObservation
from rift.gym_carla.observation.ft_pluto_observation import CBVGRPOPlutoObservation, CBVRSPlutoObservation, CBVRTRPlutoObservation, CBVReinforcePlutoObservation, CBVPPOPlutoObservation, CBVSFTPlutoObservation

OBSERVATION_LIST = {
    'ego_no_obs': EgoBaseObservation,
    'cbv_no_obs': CBVBaseObservation,
    'ego_simple_obs': EgoSimpleObservation,
    'ego_normal_obs': EgoNormalObservation,
    'ego_e2e_obs': EgoBaseObservation,
    'cbv_normal_obs': CBVNormalObservation,
    'cbv_pluto_obs': CBVPlutoObservation,
    'ego_pluto_obs': EgoPlutoObservation,
    'cbv_sft_pluto_obs': CBVSFTPlutoObservation,
    'cbv_rtr_pluto_obs': CBVRTRPlutoObservation,
    'cbv_rs_pluto_obs': CBVRSPlutoObservation,
    'cbv_reinforce_pluto_obs': CBVReinforcePlutoObservation,
    'cbv_ppo_pluto_obs': CBVPPOPlutoObservation,
    'cbv_grpo_pluto_obs': CBVGRPOPlutoObservation,
}
