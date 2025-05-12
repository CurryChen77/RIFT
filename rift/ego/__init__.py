#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2023/10/4
"""

# for planning scenario

from rift.ego.b2d.e2e_agent import VAD, UniAD
from rift.ego.rl.ppo import PPO
from rift.ego.behavior import CarlaBehaviorAgent
from rift.ego.expert_disturb import CarlaExpertDisturbAgent
from rift.ego.expert.expert import CarlaExpertAgent
from rift.ego.plant.plant import PlanT
from rift.ego.pdm_lite.pdm_lite import PDM_LITE


EGO_POLICY_LIST = {
    'behavior': CarlaBehaviorAgent,
    'ppo': PPO,
    'expert': CarlaExpertAgent,
    'plant': PlanT,
    'expert_disturb': CarlaExpertDisturbAgent,
    'pdm_lite': PDM_LITE,
    'vad': VAD,
    'uniad': UniAD,
}
