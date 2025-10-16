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


class BaseSentinelAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 detect_interval=-1, patrol_config=None):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.looking_down = False
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger, knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json"))

        self.spot_counter = dict()
        self.visible_agent = dict()
        self.countdown = 0
        self.current_target = None

        self.patrol_config = patrol_config

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
        tmp_arr=set(self.obs['segmentation'].flatten().tolist())
        values, counts = np.unique(self.obs['segmentation'], return_counts=True)
        freq = dict(zip(values, counts))
        self.visible_agent = {}
        for i in freq:
            if freq[i] < 0.0002 * len(self.obs['segmentation'].flatten()): continue
            e = self.obs["gt_seg_entity_idx_to_info"][i]
            if 'type' in e and e['type'] == 'avatar': # e[-1] is None
                if 'Sentinel' in e['name']: continue
                if e['name'] not in self.obs['agent_pos_dict']: continue
                self.visible_agent[e['name']] = freq[i]
                if e['nameT'] not in self.spot_counter:
                    self.spot_counter[e['name']] = 0
                self.spot_counter[e['name']] += 1
                mask = (self.obs['segmentation'] == i)
                selected_depths = self.obs['depth'][mask]
                depth = np.median(selected_depths)
                self.logger.info(f"I see {e['name']}. its position is at {self.obs['agent_pos_dict'][e['name']]}, our distance is {np.linalg.norm(np.array(self.pose[:2]) - np.array(self.obs['agent_pos_dict'][e['name']]['pose'][:2]))} and the median depth is {depth}. The frequency is {freq[i]}")

    def _act(self, obs):
        if len(self.visible_agent) == 0:
            self.set_target(None)
            action = self.patrol()
        else:
            if self.current_target is not None and self.current_target in self.visible_agent:
                self.countdown -= 1 if self.visible_agent[self.current_target]<0.0005 * len(self.obs['segmentation'].flatten()) else 2
                if self.countdown > 0:
                    action = {"type": "wait"}
                else:
                    action = {"type": "signal", "arg1": f"ban", "arg2": self.current_target}
                    self.set_target(None)
            else:
                for agent_name in self.visible_agent:
                    self.set_target(agent_name)
                    action = {"type": "signal", "arg1": f"warning", "arg2": agent_name}
                    break
        
        self.last_action=action
        return self.last_action
    
    def set_target(self, agent_name):
        if agent_name is None:
            self.current_target = None
            self.countdown = 0
        else:
            self.current_target = agent_name
            self.countdown = 15
    
    def patrol(self):
        if self.patrol_config is None:
            return None
        if self.patrol_config["type"] == "fixed":
            return {"type": "wait"}
        elif self.patrol_config["type"] == "rotating":
            return {"type": "turn_right", "arg1": 30}
        elif self.patrol_config["type"] == "patrolling":
            while is_near_goal(curr_x=self.pose[0], curr_y=self.pose[1], goal_bbox=None, goal_pos=self.patrol_config["route"][self.patrol_config["route_index"]]):
                self.patrol_config["route_index"]=(self.patrol_config["route_index"]+1)%(len(self.patrol_config["route"]))
            action = self.navigate(self.s_mem.get_sg(), goal_pos=self.patrol_config["route"][self.patrol_config["route_index"]], goal_bbox=None)
            return action
