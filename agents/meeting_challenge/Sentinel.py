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


class SentinelMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone)
        self.emergency = 0

    def reset(self, name, pose):
        super().reset(name, pose)

    def _process_obs(self, obs):
        for i in range(len(self.known_sentinel_poses)):
            if self.known_sentinel_poses[i][3]==-1:
                if obs['action_status'] == "FAIL":
                    self.known_sentinel_poses[i][3]=0
                else:
                    self.known_sentinel_poses[i][3]=1
        super()._process_obs(obs)
        emergency = 0
        for sentinel in self.visible_sentinels:
            if np.linalg.norm(np.array(self.pose[:2])-np.array(self.visible_sentinels[sentinel][:2]))<18:
                emergency = 1
        if len(obs['events']) > 0 and emergency == 0:
            for event in obs['events']:
                if event['type'] == 'signal':
                    emergency = 6
        if self.emergency == 0:
            self.emergency = emergency
        elif 1 <= self.emergency <=5: # if in emergency
            if emergency == 1: # if see sentinel
                self.emergency = emergency # restart emergency
            else: # if no sentinel seen
                self.emergency += 1 # progress the emergency
        else: # if after emergency
            if emergency == 1: # if see emergency
                self.emergency = 1 # restart emergency
            else: # if no sentinel seen
                self.emergency = (self.emergency + 1)%9 # progress the post-emergency

    def _act(self, obs):
        if self.banned:
            if self.pose[0]>-1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        # if still in emergency
        if 1 <= self.emergency <= 5:
            emergency_avoidance = self.emergency_avoid()
            if emergency_avoidance is None:
                self.logger.warning(f"I cannot find a suitable avoidance!")
                self.last_action = None
                return None
            else:
                self.logger.info(f"performing emergency avoiding. Target is {emergency_avoidance}")
                self.last_action = self.navigate(self.s_mem.get_sg(), list(emergency_avoidance))
                return self.last_action
        elif self.emergency > 5:
            self.last_action = {'type': 'turn_right', 'arg1': 90}
            return self.last_action
        # no emergency
        if any([sentinel[3]==0 for sentinel in self.known_sentinel_poses]):
            speech = f"I saw sentinel(s) at {[sentinel[:3] for sentinel in self.known_sentinel_poses if sentinel[3]==0]}"
            self.last_action = {"type": "converse", "arg1": speech, "arg2": 3200}
            for i in range(len(self.known_sentinel_poses)):
                if self.known_sentinel_poses[i][3]==0:
                    self.known_sentinel_poses[i][3]=-1
            return self.last_action
        # no sentinel to report
        self.logger.debug(f"Current mode is {self.mode}, while the trigger is {self.discussion_trigger}, mode_time_counter is {self.mode_time_counter}")
        action = None
        try:
            if self.mode is None:
                self.enter_discussion_mode(trigger="TASK START")
            if self.mode == NavAgentState.DISCUSS:
                self.mode_time_counter += 1
                if self.mode_time_counter > 80:
                    action = {"type": "task_terminate"}
                    self.logger.info(f"Exceeding discussion limit. Task terminating.")
                    return action
                action = self.discuss()
            elif self.mode == NavAgentState.NAVIGATE:
                self.mode_time_counter += 1
                if self.meeting_place not in self.s_mem.get_places():
                    action = {"type": "query_app", "arg1": "query_place", "arg2":self.meeting_place}
                else:
                    action, arrived = self.city_navigate(self.meeting_place, rethink=True, requery=False)
                    if arrived:
                        action = {'type': 'task_complete'}
        except Exception as e:
            self.logger.error(f"Error in action generation: {e} with traceback: {traceback.format_exc()}. The plan was {action}")
            action = None
        self.action_history.append(Action(action, self.curr_time, self.curr_time))
        self.logger.debug(f"{self.name}'s current generated action is {action}.")
        assert action is None or isinstance(action, dict)
        self.last_action=action
        return self.last_action
    
    def emergency_avoid(self):
        near_sentinels = []
        for sentinel_pose in self.known_sentinel_poses:
            if np.linalg.norm(np.array(self.pose[:2])-np.array(sentinel_pose[:2])) < 20:
                near_sentinels.append(sentinel_pose)
        # get occ map
        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder
        occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map() # occ map: 1 for unknow, 2 for obstacle, 3 for open
        def valid(x, y):
            px, py = builder.align_nav(x)-x_min, builder.align_nav(y)-y_min
            return 0 <= int(py) < y_max - y_min and 0 <= int(px) < x_max - x_min and occ_map[int(py)][int(px)] not in [2, 4]
        valid_pos = []
        for x in range(int(self.pose[0])-40, int(self.pose[0])+40):
            for y in range(int(self.pose[1])-40, int(self.pose[1])+40):
                if not valid(x, y): continue
                value = 0
                for sentinel_pose in near_sentinels:
                    value += (x - self.pose[0])*(sentinel_pose[0]-self.pose[0]) + (y - self.pose[1])*(sentinel_pose[1]-self.pose[1])
                if value > 0:
                    valid_pos.append((value, x, y))
        if not valid_pos:
            return None  # no safe direction

        # convert to probabilities that prefer moving away from sentinels
        values = np.array([v for v, _, _ in valid_pos])
        # invert and normalize to make lower values (away from sentinels) more probable
        weights = np.exp(-values / (np.std(values) + 1e-6))
        probs = weights / np.sum(weights)

        # random weighted selection
        idx = np.random.choice(len(valid_pos), p=probs)
        _, target_x, target_y = valid_pos[idx]

        print(f"[Avoidance] Selected new target ({target_x:.2f}, {target_y:.2f})")

        # Return or update navigation goal
        return (target_x, target_y)