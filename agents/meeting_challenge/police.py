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
from ViCo.tools.utils import *
from ViCo.tools.model_manager import global_model_manager
from agents.sg.builder.builder import Builder, BuilderConfig


class AdversaryAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 route=None, detect_interval=-1):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.route=route
        self.route_index=0
        self.looking_down = False
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger, knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json"))

    def reset(self, name, pose):
        super().reset(name, pose)
        self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)

    def _process_obs(self, obs):
        num_new_objects = self.s_mem.update(obs)
        self.curr_time = obs['curr_time']
        self.held_objects = obs['held_objects']
        self.current_place = obs['current_place']
        self.obs = obs

    def _act(self, obs):
        while is_near_goal(curr_x=self.pose[0], curr_y=self.pose[1], goal_bbox=None, goal_pos=self.route[self.route_index]):
            self.route_index=(self.route_index+1)%(len(self.route))
        action = self.navigate(self.s_mem.get_sg(), goal_pos=self.route[self.route_index], goal_bbox=None)
        self.last_action=action
        return self.last_action
