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


@dataclass
class Place:
    name: str
    location: list[float, float] | None = None
    bbox: list[float, float, float, float] | None = None
    region: dict | None = None

    def __init__(self, target, region_name, s_mem):
        if type(target) == dict:
            # region
            target_region = target['region']
            self.name = region_name
            target_pos = [(target_region['x_min'] + target_region['x_max']) / 2,
                      (target_region['y_min'] + target_region['y_max']) / 2]
            self.location = round_numericals(target_pos)
            self.bbox = round_numericals([target_region['x_min'], target_region['y_min'],
                         target_region['x_max'], target_region['y_max']])
            self.region = target['region']
            return
        self.name = target
        if target in s_mem.get_places():
            # place
            place_dict = s_mem.get_knowledge(target)
            self.location = round_numericals([place_dict["location"][0] - 1000, place_dict["location"][1] - 1000])
            bbox = place_dict["bounding_box"]
            if bbox is None:
                # outdoor place
                self.bbox = [self.location[0] - 4, self.location[1] - 4, self.location[0] + 4, self.location[1] + 4]
            else:
                self.bbox = round_numericals(bbox3d_to_bbox2d(bbox_center_to_corners_repr(bbox)))
            return
        # agent

    def within(self, point: list[float, float]) -> bool:
        if self.region is not None:
            return (self.region['x_min'] <= point[0] <= self.region['x_max'] and
                    self.region['y_min'] <= point[1] <= self.region['y_max'])
        elif self.bbox is not None:
            return (self.bbox[0] <= point[0] <= self.bbox[2] and
                    self.bbox[1] <= point[1] <= self.bbox[3]) or \
                     (self.bbox[0] - 1000 <= point[0] <= self.bbox[2] - 1000 and
                        self.bbox[1] - 1000 <= point[1] <= self.bbox[3] - 1000) or \
                        (self.bbox[0] + 1000 <= point[0] <= self.bbox[2] + 1000 and
                        self.bbox[1] + 1000 <= point[1] <= self.bbox[3] + 1000)
        else:
            return False


@dataclass
class Action:
    action: dict
    start_time: datetime
    end_time: datetime

    def to_description(self):
        action_to_print = copy.deepcopy(self.action)
        if "arg2" in action_to_print:
            action_to_print.pop("arg2")
        if self.action["type"] == "converse":
            action_to_print.pop("arg1")
        return f"{self.start_time.strftime('%H:%M:%S')} - {self.end_time.strftime('%H:%M:%S') if self.end_time else ''}: {action_to_print}"

    def judge_continue(self, current_plan):
        if self.action["type"] == "converse" and current_plan["type"] == "converse":
            return True
        return self.action == current_plan and self.action["type"] not in ["put", "pick"]


@dataclass
class Chat:
    time: datetime
    subject: str
    content: str

    def to_description(self):
        return f"{self.time.strftime('%H:%M:%S')} {self.subject}: {self.content}"

class LLMMeetingAgent(Agent):
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

        self.action_history: list[Action] = []
        self.current_plan = None
        self.plan_start_time = None
        self.conversation_history: list[Chat] = []
        self.meeting_place = None

    def reset(self, name, pose):
        super().reset(name, pose)
        self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), debug=self.debug, logger=self.logger)
        self.meeting_place = None

    def _process_obs(self, obs):
        if obs['action_status'] == "FAIL":
            self.logger.info(f"{self.name} failed to execute last action {self.action_history[-1].action}.")
            if self.action_history[-1].action["type"] == "converse":
                if len(self.conversation_history) > 0 and self.conversation_history[-1].subject == self.name:
                    self.conversation_history.pop()
        if len(obs['events']) > 0:
            for event in obs['events']:
                if event["type"] == "speech":
                    if event["subject"] == self.name:
                        continue
                    self.conversation_history.append(Chat(self.curr_time, event["subject"], event["content"]))
        num_new_objects = self.s_mem.update(obs)
        self.curr_time = obs['curr_time']
        self.held_objects = obs['held_objects']
        self.current_place = obs['current_place']
        self.obs = obs

    def _act(self, obs):
        action = None
        try:
            if self.meeting_place == None:
                response_type, speech = self.get_meeting_place()
                if response_type is None or response_type == "wait":
                    action = {"type": "wait"}
                elif response_type == "speak":
                    action = {"type": "speech", "arg1": speech, "arg2": 800}
                    self.conversation_history.append(Chat(self.curr_time + timedelta(seconds=1), self.name, action['arg1']))
                elif response_type == "decide":
                    self.meeting_place = speech
                    action = {"type": "wait"}
                else:
                    raise NotImplementedError(f"meeting place response type {response_type} is not supported")
            else:
                action = self.goto_place(self.meeting_place)
                arrived = self.current_place == self.meeting_place
                if arrived:
                    action = {'type': 'task_complete'}
        except Exception as e:
            self.logger.error(f"Error in action generation: {e} with traceback: {traceback.format_exc()}. The plan was {action}")
            action = None
        return action
    
    def get_meeting_target(self):
        # use this function to get geometric center
        meeting_target = np.zeros(2, dtype=float)
        for agent in self.obs['agent_pos_dict']:
            meeting_target += np.array(self.obs['agent_pos_dict'][agent]['pose'][:2])
        meeting_target /= len(self.obs['agent_pos_dict'])
        return meeting_target
    
    def get_meeting_place(self):
        prompt = open(f"agents/meeting_challenge/meeting_prompts/get_meeting_place_prompt.txt", "r").read()
        prompt = prompt.replace("$SelfName$", self.name)
        agent_pos_dict=copy.copy(self.obs["agent_pos_dict"])
        agent_pos_description = ""
        for agent in agent_pos_dict:
            if agent_pos_dict[agent]['place'] is not None:
                agent_pos_dict[agent]['pose'][0], agent_pos_dict[agent]['pose'][1] = agent_pos_dict[agent]['pose'][0]-1000, agent_pos_dict[agent]['pose'][1]-1000
            agent_pos_description += f"{agent} is now in {agent_pos_dict[agent]['place'] if agent_pos_dict[agent]['place'] is not None else 'open space'}, with coordinate {agent_pos_dict['pose']}.\n"
        agent_pos_description.strip("\n")
        prompt = prompt.replace("$AgentPoses$", agent_pos_description)
        prompt = prompt.replace("$Places$", self.get_nearest_places_description(self.get_meeting_target()))
        prompt = prompt.replace("$ConversationHistory$", self.conversation_history)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
            meeting_place = response_dict['meeting_place']
            speech = response_dict['speech']
        except Exception as e:
            self.logger.error(
                f"Error getting meeting place: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            meeting_place = None
            speech = None
        return meeting_place, speech

    def parse_json(self, prompt, response, last_call=False):
        json_str = None
        if "```json" in response:
            # Step 1: Extract the JSON part
            start = response.find("```json") + len("```json")
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            self.logger.warning(f"Error parsing JSON, the string was {response}")
            if not last_call:
                chat_history = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                data = self.generator.generate(
                    f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!",
                    chat_history=chat_history)
                return self.parse_json(None, data, last_call=True)
            else:
                self.logger.error(f"Error parsing JSON, already last call, the string was {response}")
                return None

        # # Step 2: Clean up the JSON
        # # Replace single quotes with double quotes
        # # Safely evaluate the string to a Python dictionary
        # parsed_dict = ast.literal_eval(json_str)
        # # Convert the dictionary back to a JSON string
        # json_str = json.dumps(parsed_dict)

        # Step 3: Convert to dictionary
        try:
            response = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding JSON: {e}, the string was {json_str}")
            if not last_call:
                chat_history = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                data = self.generator.generate(
                    f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!",
                    chat_history=chat_history)
                return self.parse_json(None, data, last_call=True)
        return response

    def search_nearby(self, source=None):
        '''
            search_range: [x_min, x_max, y_min, y_max]
        '''
        search_range = None
        self.logger.debug(f"Searching {self.search_target}")
        if source is not None:
            if type(source) is dict:
                search_range = [source['region']['x_min'], source['region']['x_max'],
                                source['region']['y_min'], source['region']['y_max']]
            elif type(source) is str and source in self.s_mem.get_places():
                knowledge = self.s_mem.get_knowledge(source)
                bbox = knowledge.get('bounding_box', None)
                if bbox is not None:
                    bbox = bbox_center_to_corners_repr(bbox)
                    bbox = irregular_to_regular_bbox(bbox)
                    search_range = [np.min(bbox[:, 0]), np.max(bbox[:, 0]),
                                    np.min(bbox[:, 1]), np.max(bbox[:, 1])]
        if not self.looking_down:
            self.looking_down = True
            return {'type': 'look_down'}
        reach_target_distance = 2. if self.current_place is None else 1.
        if self.search_target is not None:
            if self.search_target[0] - reach_target_distance < self.pose[0] < self.search_target[0] + reach_target_distance and \
                    self.search_target[1] - reach_target_distance < self.pose[1] < self.search_target[1] + reach_target_distance:
                # search target has been reached
                self.search_target = None
            elif search_range is not None and (not search_range[0] < self.search_target[0] < search_range[1] or not \
                    search_range[2] < self.search_target[1] < search_range[3]):
                # search target is out of the box
                self.search_target = None
            else:
                return self.navigate(self.s_mem.get_sg(self.current_place), self.search_target)

        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder

        if self.current_place is None:
            # find nearest unexplored point
            occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map()
            agent_pos_in_map = [builder.align_nav(self.pose[0]) - x_min,
                                builder.align_nav(self.pose[1]) - y_min]
            rows, cols = np.where(occ_map == 1)
            dists = np.sqrt((rows - agent_pos_in_map[0]) ** 2 + (cols - agent_pos_in_map[1]) ** 2)
            xs = [(row + x_min) * builder.conf.nav_grid_size for row in rows]
            ys = [(col + y_min) * builder.conf.nav_grid_size for col in cols]
            order = np.argsort(dists)
            sorted_rows = rows[order]
            sorted_cols = cols[order]

            if search_range is not None:
                mask = []
                bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = search_range
                for x, y in zip(xs, ys):
                    mask.append(bbox_x_min <= x <= bbox_x_max and bbox_y_min <= y <= bbox_y_max)
                self.logger.debug(f"Search mask bbox {[bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max]}")
                self.logger.debug(f"Search mask size {sum(mask)}")
                if sum(mask) > 0:
                    sorted_rows = sorted_rows[mask]
                    sorted_cols = sorted_cols[mask]

            sorted_rows = sorted_rows[:100]
            sorted_cols = sorted_cols[:100]
            positions = list(zip(sorted_rows, sorted_cols))
            chosen_position = None
            if len(positions) > 0:
                row, col = random.choice(positions)
                x = (row + x_min) * builder.conf.nav_grid_size
                y = (col + y_min) * builder.conf.nav_grid_size
                chosen_position = [x, y]
            else:
                self.logger.error("Can not find a search target!")
            self.search_target = chosen_position
        else:
            self.search_target = None
        if self.search_target is not None:
            return self.navigate(self.s_mem.get_sg(self.current_place), self.search_target)
        else:
            return {"type": "turn_left",
                    "arg1": 90}

    def get_nearest_places_description(self, target):
        place_list = []
        for place in self.s_mem.get_places():
            goal_place_dict = self.s_mem.get_knowledge(place)
            if goal_place_dict is None:
                self.logger.error(f"No knowledge found for {place}.")
                return None, False
            goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
            if goal_place_dict["building"] != "open space":
                goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
            goal_bbox = goal_place_dict["bounding_box"]
            place_list.append((np.linalg.norm(np.array(target)-goal_pos),place))
        place_list = sorted(place_list)
        place_list = place_list[:15] if len(place_list)>15 else place_list
        places_description = ""
        for dis, place in place_list:
            goal_place_dict = self.s_mem.get_knowledge(place)
            if goal_place_dict is None:
                self.logger.error(f"No knowledge found for {place}.")
                return None, False
            goal_pos = np.array([goal_place_dict["location"][0], goal_place_dict["location"][1]])
            if goal_place_dict["building"] != "open space":
                goal_pos[0], goal_pos[1] = goal_pos[0] - 1000, goal_pos[1] - 1000
            goal_bbox = goal_place_dict["bounding_box"]
            places_description += f"<{place}>: location {goal_pos}, bounding box {goal_bbox}\n"
        return places_description

    def get_previous_actions_description(self):
        if len(self.action_history) == 0:
            return "None"
        else:
            action_list = self.action_history[-10:] if len(self.action_history) > 10 else self.action_history
            return "\n".join([action.to_description() for action in action_list])

    def get_conversation_description(self):
        if len(self.conversation_history) == 0:
            return "None"
        conversation_list = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        return "\n".join([chat.to_description() for chat in conversation_list])

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