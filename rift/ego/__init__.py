#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2023/10/4
"""

# for planning scenario

from rift.ego.b2d.e2e_agent import VAD, SparseDrive, UniAD
from rift.ego.rl.ppo import PPO
from rift.ego.behavior import Behavior
from rift.ego.expert_disturb import ExpertDisturb
from rift.ego.expert.expert import Expert
from rift.ego.plant.plant import PlanT
from rift.ego.pdm_lite.pdm_lite import PDM_LITE


EGO_POLICY_LIST = {
    'behavior': Behavior,
    'ppo': PPO,
    'expert': Expert,
    'plant': PlanT,
    'expert_disturb': ExpertDisturb,
    'pdm_lite': PDM_LITE,
    'vad': VAD,
    'uniad': UniAD,
    'sparsedrive': SparseDrive,
}
