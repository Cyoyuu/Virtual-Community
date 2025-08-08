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
from sg.builder.builder import Builder, BuilderConfig


class HeuristicMeetingAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.looking_down = False
        self.num_agents = num_agents
        self.comm = self.num_agents > 1
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger)

        if init_generator:

            self.generator = global_model_manager.get_generator(lm_source, lm_id, max_tokens, temperature, top_p, logger)
        else:
            self.generator = None

        self.end_time = None

        self.meeting_target = None
        self.stage = 0

    def reset(self, name, pose):
        super().reset(name, pose)
        self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)
        self.meeting_target = None

    def _process_obs(self, obs):
        num_new_objects = self.s_mem.update(obs)
        self.found_new_objects = False
        self.curr_time = obs['curr_time']
        self.held_objects = obs['held_objects']
        self.current_place = obs['current_place']
        self.obs = obs

    def _act(self, obs):
        action = None
        if self.stage == 0:
            self.meeting_target = self.get_meeting_target()
            action = self.navigate(self.s_mem.get_sg(self.current_place), self.meeting_target, goal_bbox=None)
            arrived = is_near_goal(self.pose[0], self.pose[1], None, self.meeting_target)
            if arrived:
                self.stage = 10
            else:
                self.stage += 1
        elif self.stage <= 9:
            action = self.navigate(self.s_mem.get_sg(self.current_place), self.meeting_target, goal_bbox=None)
            arrived = is_near_goal(self.pose[0], self.pose[1], None, self.meeting_target)
            if arrived:
                self.stage = 10
            else:
                self.stage = (self.stage+1)%10
        elif self.stage == 10:
            self.meeting_target = self.get_meeting_target()
            action = self.navigate(self.s_mem.get_sg(self.current_place), self.meeting_target, goal_bbox=None)
            arrived = is_near_goal(self.pose[0], self.pose[1], None, self.meeting_target, threshold=5)
            if arrived:
                action = {'type': 'task_complete'}
            else:
                self.stage = 1
        return action

    def get_meeting_target(self):
        # use this function to get geometric center
        meeting_target = np.zeros(2, dtype=float)
        for agent in self.obs['agent_pos_dict']:
            meeting_target += np.array(self.obs['agent_pos_dict'][agent]['pose'][:2])
        meeting_target /= len(self.obs['agent_pos_dict'])
        return meeting_target

    def goto(self, target, force=False):
        if target.startswith("task_"):
            # it is a region
            _, task_idx, source, _ = target.split("_")
            task_idx = int(task_idx)
            if source == "source":
                target = self.task_manager.tasks[task_idx].source[0]
            else:
                target = self.task_manager.tasks[task_idx].destination[0]
            return self.goto_region(target['region'], force=force)
        if type(target) is str:
            # it is a room or agent
            if target in self.get_agent_list():
                return self.goto_agent(target, force=force)
            else:
                return self.goto_place(target, force=force)
        else:
            # it is a region
            return self.goto_region(target['region'], force=force)
    
    def get_agent_list(self):
        return self.s_mem.agents

    def get_indoor_places(self):
        return self.s_mem.get_indoor_places()

    def get_outdoor_places(self):
        return self.s_mem.get_outdoor_places()

    def goto_place(self, target_place: str, force=False) -> (dict, bool):
        places = self.get_indoor_places() + self.get_outdoor_places() + ['open space']
        if target_place is None:
            target_place = 'open space'
            self.logger.debug(
                f"{self.name} at {self.obs['current_place']} is entering open space.")
            self.last_action = {
                'type': 'enter',
                'arg1': 'open space'
            }
            return self.last_action, True
        self.logger.info(f"Currently goto_place {target_place}.")
        if target_place not in places:
            self.logger.error(f"Target place {target_place} is not a valid place.")
            return None, False
        if force:
            self.last_action = {'type': 'force_enter', 'arg1': target_place}
            return self.last_action, True
        goal_place_dict = self.s_mem.get_knowledge(target_place)
        if goal_place_dict is None:
            self.logger.error(f"No knowledge found for {target_place}.")
            return None, False
        goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
        if goal_place_dict["building"] != "open space":
            goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
        goal_bbox = goal_place_dict["bounding_box"]
        self.logger.debug(f"Goal place: {target_place}, goal pos: {goal_pos}, goal bbox: {goal_bbox}")
        self.last_action = {'type': 'wait', 'arg1': None}
        # already at the correct place
        if target_place == self.obs['current_place']:
            self.logger.debug(f"{self.name} arrived at {target_place}.")
            return self.last_action, True
        # can enter the correct place
        if target_place in self.obs['accessible_places']:
            self.logger.debug(f"{self.name} finished navigation to {target_place} at {goal_pos}")
            self.last_action = {
                'type': 'enter',
                'arg1': target_place
            }
            return self.last_action, True
        # at wrong place, need to enter open space
        if self.obs['current_place'] is not None:
            self.logger.debug(
                f"{self.name} at {self.obs['current_place']} is entering open space to move to {target_place} at {goal_pos}.")
            self.last_action = {
                'type': 'enter',
                'arg1': 'open space'
            }
            return self.last_action, False
        # at open space, need to move to the correct place
        cur_trans = np.array(self.pose[:2])
        if is_near_goal(cur_trans[0], cur_trans[1], goal_bbox, goal_pos):
            self.logger.warning(
                f"{self.name} at {self.pose} is near the goal {goal_pos}, but not at the goal {target_place}.")
            return self.last_action, True
        self.logger.debug(
            f"{self.name} at {tuple(int(p) for p in self.pose)} is moving to {target_place} at {tuple(int(g) for g in goal_pos)}.")
        start = time.time()
        self.last_action = self.navigate(self.s_mem.get_sg(self.current_place), goal_pos, goal_bbox)
        self.logger.debug(f"Navigate time: {start}, {time.time()}")
        return self.last_action, False

    def goto_region(self, target_region: dict, force=False):
        if self.current_place is not None:
            return {
                'type': 'enter',
                'arg1': 'open space'
            }, False
        target_pos = [(target_region['x_min'] + target_region['x_max']) / 2,
                      (target_region['y_min'] + target_region['y_max']) / 2]
        if force:
            return {
                'type': 'teleport',
                'arg1': target_pos
            }, True
        action = self.navigate(self.s_mem.get_sg(self.current_place), target_pos, goal_bbox=None)
        arrived = False
        if target_region['x_min'] < self.pose[0] < target_region['x_max'] and \
                target_region['y_min'] < self.pose[1] < target_region['y_max']:
            arrived = True
        return action, arrived