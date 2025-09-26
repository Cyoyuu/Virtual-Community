import ast
import os
import pdb
import random
import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
import pickle
import re
from enum import Enum
import time

from agents.agent import Agent
from agents.memory import SemanticMemory
from agents.meeting_challenge.base_nav import *
from ViCo.tools.utils import *
from ViCo.tools.model_manager import global_model_manager
from agents.sg.builder.builder import Builder, BuilderConfig


class HeuristicNavigationMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents)

    def reset(self, name, pose):
        super().reset(name, pose)

    # def _process_obs(self, obs):

    def _act(self, obs):
        self.logger.debug(f"self mode time counter is {self.mode_time_counter}")
        action = None
        if self.mode_time_counter % 20 == 0:
            self.meeting_place = self.get_meeting_place()
            self.enter_navigation_mode()
        action, arrived = self.city_navigate(self.meeting_place)
        if arrived:
            action = {'type': 'task_complete'}
        self.mode_time_counter +=1
        self.last_action = action
        return action