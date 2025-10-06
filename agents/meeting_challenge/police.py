from agents.meeting_challenge.patrol_utils import generate_random_patrol_route

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
        self.route = route
        self.route_index = 0
        self.looking_down = False
        self.s_mem = SemanticMemory(
            os.path.join(self.storage_path, "semantic_memory"),
            detect_interval=detect_interval,
            debug=self.debug,
            logger=self.logger,
            knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json")
        )

        self.spot_counter = dict()
        self.visible_agent = list()

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
        tmp_arr = set(self.obs['segmentation'].flatten().tolist())
        self.visible_agent = list()
        for i in tmp_arr:
            e = self.obs["gt_seg_entity_idx_to_info"][i]
            self.logger.info(f"gt seg {i} is {e}")
            if 'type' in e and e['type'] == 'avatar':
                self.visible_agent.append(e['name'])
                if e['name'] not in self.spot_counter:
                    self.spot_counter[e['name']] = 0
                self.spot_counter[e['name']] += 1

    def _act(self, obs):
        # --- new added ---
        if not self.route:
            try:
                self.route = generate_random_patrol_route(
                    amap=obs.get("nav_app") if isinstance(obs, dict) else None,
                    current_xy=(self.pose[0], self.pose[1]),
                    n_points=12,
                    min_hop=60.0,
                    max_hop=200.0
                )
                self.route_index = 0
                self.logger.info(f"[{self.name}] Generated random patrol route with {len(self.route)} waypoints.")
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to generate random patrol route: {e}")
                self.route = [(self.pose[0], self.pose[1])]
        # --- new added ---

        for agent_name in self.visible_agent:
            self.logger.info(f"I see {agent_name}.")

        while is_near_goal(
            curr_x=self.pose[0],
            curr_y=self.pose[1],
            goal_bbox=None,
            goal_pos=self.route[self.route_index]
        ):
            self.route_index = (self.route_index + 1) % len(self.route)

        action = self.navigate(
            self.s_mem.get_sg(),
            goal_pos=self.route[self.route_index],
            goal_bbox=None
        )
        self.last_action = action
        return self.last_action
