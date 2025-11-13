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


class RoCoMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone)

    def reset(self, name, pose):
        super().reset(name, pose)

    # def _process_obs(self, obs):

    def _act(self, obs):
        if self.banned:
            if self.pose[0]>-1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        self.logger.debug(f"Current mode is {self.mode}, while the trigger is {self.discussion_trigger}")
        action = None
        try:
            if self.mode is None:
                self.enter_discussion_mode(trigger="TASK START")
            if self.mode == NavAgentState.DISCUSS:
                self.mode_time_counter += 1
                if self.mode_time_counter > 60:
                    action = {"type": "task_terminate"}
                    self.logger.info(f"Exceeding discussion limit. Task terminating.")
                    return action
                action = self.discuss_act()
            elif self.mode == NavAgentState.NAVIGATE:
                self.mode_time_counter += 1
                if self.meeting_place not in self.s_mem.get_places():
                    action = {"type": "query_app", "arg1": "query_place", "arg2":self.meeting_place}
                else:
                    action, arrived = self.city_navigate(self.meeting_place)
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