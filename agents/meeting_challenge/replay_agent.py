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
import math

from agents.agent import Agent
from agents.memory import SemanticMemory
from agents.sg.builder.builder import Builder, BuilderConfig
from tools.utils import *


class ReplayAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 detect_interval=-1, steps=None):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.looking_down = False
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger, knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json"))

        self.steps = steps if steps is not None else []  # @ruxi fill this: action list read from the output
        self.step = 0

    def reset(self, name, pose):
        super().reset(name, pose)
        self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)
        self.step = 0

    def _process_obs(self, obs):
        pass

    def _act(self, obs):
        if self.steps is None or len(self.steps) == 0:
            return {'type': 'task_terminate'}

        if self.step >= len(self.steps):
            return {'type': 'task_terminate'}

        action = self.steps[self.step]
        self.step += 1

        if isinstance(action, str):
            if action in ['move_forward', 'turn_left', 'turn_right', 'enter', 'enter_bus', 'exit_bus']:
                # @ruxi fill this: execute action
                return {'type': action}
            else:
                return {'type': 'wait'}

        return {'type': 'wait'}
