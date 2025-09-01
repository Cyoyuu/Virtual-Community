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


class NavAgentState(Enum):
    DISCUSS    = "BASE_DISCUSSING"
    NAVIGATE  = "BASE_NAVIGATING"

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

class NavigationMeetingAgent(Agent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)
        self.looking_down = False
        self.num_agents = num_agents
        self.comm = self.num_agents > 1
        self.s_mem = SemanticMemory(os.path.join(self.storage_path, "semantic_memory"), detect_interval=detect_interval, debug=self.debug, logger=self.logger, knowledge_path=os.path.join(self.storage_path, "seed_knowledge.json"))

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
        self.mode = None
        self.last_estimated_arrival_time = None
        self.last_route = []
        self.last_nav = []
        self.last_action = None
        self.route_history = dict()

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
                if event["type"] == "message":
                    pass
                if event["type"] == "app message":
                    if self.last_action['type']=="query_app" and self.last_action['arg1']=="query_route":
                        self.last_route=event["content"]
        num_new_objects = self.s_mem.update(obs)
        self.curr_time = obs['curr_time']
        self.held_objects = obs['held_objects']
        self.current_place = obs['current_place']
        self.obs = obs
        if self.obs['steps']%100==0:
            self.route_history[self.obs['steps']]=copy.deepcopy(self.last_route)
            json.dump(self.route_history, open(os.path.join(self.storage_path, "route_history.json"), "w"))

    def _act(self, obs):
        action = None
        try:
            if self.mode is None:
                self.mode = NavAgentState.DISCUSS
            if self.mode == NavAgentState.DISCUSS:
                response_type, speech = self.get_meeting_place()
                if response_type is None or response_type == "wait":
                    action = {"type": "wait"}
                elif response_type == "speak":
                    action = {"type": "converse", "arg1": speech, "arg2": 800}
                    self.conversation_history.append(Chat(self.curr_time + timedelta(seconds=1), self.name, action['arg1']))
                elif response_type == "decide":
                    self.meeting_place = speech
                    action = {"type": "wait"}
                    self.mode = NavAgentState.NAVIGATE
                else:
                    raise NotImplementedError(f"meeting place response type {response_type} is not supported")
            elif self.mode == NavAgentState.NAVIGATE:
                action, arrived = self.city_navigate(self.meeting_place)
                if arrived:
                    action = {'type': 'task_complete'}
                    self.logger.info(f"Currently arrived at {self.meeting_place}.")
        except Exception as e:
            self.logger.error(f"Error in action generation: {e} with traceback: {traceback.format_exc()}. The plan was {action}")
            action = None
        self.action_history.append(Action(action, self.curr_time, self.curr_time))
        self.logger.debug(f"{self.name}'s current generated action is {action}.")
        assert action is None or isinstance(action, dict)
        self.last_action=action
        return self.last_action
    
    def city_navigate(self, goal_place, threshold=500.):
        self.logger.info(f"Currently city nav to {goal_place}. The remaining route waypoints is {len(self.last_route)}. The estimated time till arrival is {timedelta(seconds=self.calc_time())}s")
        # already at the correct place (or to comply with env setting)
        if goal_place == self.obs['current_place'] or (goal_place in self.obs['accessible_places'] and self.s_mem.get_knowledge(goal_place)["building"]=="open space"):
            self.logger.debug(f"{self.name} arrived at {goal_place}.")
            return self.last_action, True
        # can enter the correct place
        if goal_place in self.obs['accessible_places']:
            self.logger.debug(f"{self.name} finished navigation to {goal_place}")
            self.last_action = {
                'type': 'enter',
                'arg1': goal_place
            }
            return self.last_action, False
        # at wrong place, need to enter open space
        if self.obs['current_place'] is not None:
            self.logger.debug(
                f"{self.name} at {self.obs['current_place']} is entering open space to move to {goal_place}.")
            self.last_action = {
                'type': 'enter',
                'arg1': 'open space'
            }
            return self.last_action, False
        if not self.last_route:
            action = {"type": "query_app", "arg1": "query_route", "arg2": goal_place}
            return action, False
        # estimated_arrival_time = self.curr_time + calc_time()
        # if self.last_estimated_arrival_time < estimated_arrival_time + THRES:
        #     action = {"type": "query_app", "arg1": "query_route", "arg2": goal_place}
        #     return action, False
        cur_trans = np.array(self.pose[:2])
        idx = len(self.last_route)
        while idx > 0:
            idx -= 1
            arrived = is_near_goal(cur_trans[0], cur_trans[1], None, self.last_route[idx], threshold=5)
            if arrived:
                for i in range(idx+1):
                    self.last_route.pop(0)
        return self.llm_navigate(max_retry=3)
    
    def llm_navigate(self, max_retry = 3, threshold=200.):
        assert self.last_route is not None
        self.logger.debug(f"Current last_nav is {self.last_nav}")
        if not self.last_nav:
            self.generate_navigation_plan(max_retry=max_retry)
        # estimated_arrival_time = self.curr_time + calc_time()
        # if self.last_estimated_arrival_time < estimated_arrival_time + THRES:
        #     self.generate_navigation_plan(max_retry)
        arrived = True
        curr_goal = None
        cur_trans = np.array(self.pose[:2])
        while arrived:
            if not self.last_nav:
                self.generate_navigation_plan(max_retry=max_retry)
                if not self.last_nav:
                    return {"type": "wait"}, False
            curr_goal = self.last_nav[0]
            action = self.navigate(self.s_mem.get_sg(), curr_goal)
            arrived = is_near_goal(cur_trans[0], cur_trans[1], None, curr_goal, threshold=2)
            if arrived: self.last_nav.pop(0)
        return action, arrived

    def generate_navigation_plan(self, max_retry=3):
        assert max_retry >= 0
        if max_retry == 0:
            # Fallback: use first 3 waypoints from last_route
            self.last_nav = self.last_route[:min(3,len(self.last_route))]
            return
        # find nearest unexplored point
        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder
        occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map() # occ map: 1 for unknow, 2 for obstacle, 3 for open
        agent_x_world, agent_y_world = self.pose[0], self.pose[1]
        agent_pos_in_map = [
            builder.align_nav(agent_x_world) - x_min,
            builder.align_nav(agent_y_world) - y_min
        ]
        occ_map[int(agent_pos_in_map[1])][int(agent_pos_in_map[0])]=4
        # Define local crop around agent (±30m in world coordinates)
        x_low_w, x_up_w = agent_x_world - 30, agent_x_world + 30
        y_low_w, y_up_w = agent_y_world - 30, agent_y_world + 30
        # Convert to map indices
        x_low = int(max(0, builder.align_nav(x_low_w) - x_min))
        x_up = int(min(occ_map.shape[1], builder.align_nav(x_up_w) - x_min))
        y_low = int(max(0, builder.align_nav(y_low_w) - y_min))
        y_up = int(min(occ_map.shape[0], builder.align_nav(y_up_w) - y_min))

        # Crop the occupancy map
        if x_low >= x_up or y_low >= y_up:
            raise ValueError("Invalid crop bounds after alignment.")

        cropped_map = occ_map[y_low:y_up, x_low:x_up]  # Note: y first, then x

        # Function to downscale using mode (most frequent value), handling borders
        def downscale_map(map_array, factor):
            h, w = map_array.shape
            # Trim to make dimensions divisible by factor
            new_h = (h // factor) * factor
            new_w = (w // factor) * factor
            trimmed = map_array[:new_h, :new_w]
            # Reshape into blocks
            reshaped = trimmed.reshape(new_h // factor, factor, new_w // factor, factor)
            # Move block axes to the end
            blocks = reshaped.swapaxes(1, 2).reshape(new_h // factor, new_w // factor, factor * factor)
            # Apply custom priority rule per block
            downscaled = np.zeros((new_h // factor, new_w // factor), dtype=np.int32)
            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    block = blocks[i, j]
                    has_2 = 2 in block
                    has_3 = 3 in block
                    has_4 = 4 in block
                    
                    if has_2 and has_4:
                        raise ValueError(f"Block at ({i}, {j}) contains both 2 and 4, which is not allowed.")
                    elif has_4:
                        downscaled[i, j] = 4
                    elif has_2:
                        downscaled[i, j] = 2
                    elif has_3:
                        downscaled[i, j] = 3
                    else:
                        downscaled[i, j] = 1  # default to free (1)
            return downscaled

        try:
            downscaled_map = downscale_map(cropped_map, factor=4)
        except ValueError as e:
            self.logger.error(f"Downscaling failed: {e}")
            self.generate_navigation_plan(max_retry=max_retry - 1)
            return

        # Convert downscaled map to string representation
        symbol_map = {1: '.', 2: 'X', 3: '?', 4: 'A'}  # . = free, X = obstacle, ? = unknown, A = agent
        map_str_lines = []
        for row in downscaled_map:
            line = ''.join(symbol_map[val] for val in row)
            map_str_lines.append(line)
        map_str = '\n'.join(map_str_lines)

        # Convert last_route (list of [x, y] in world coords) to relative local coordinates (in cropped map, then in downscale grid)
        def world_to_downscaled_local(x_world, y_world):
            # Convert world → map index
            mx = builder.align_nav(x_world) - x_min
            my = builder.align_nav(y_world) - y_min
            # Check if in crop
            if not (x_low <= mx < x_up and y_low <= my < y_up):
                return None
            # Convert to local in cropped map
            mx_local = mx - x_low
            my_local = my - y_low
            # Downscale by 4
            mx_ds = mx_local // 4
            my_ds = my_local // 4
            if mx_ds < downscaled_map.shape[1] and my_ds < downscaled_map.shape[0]:
                return [mx_ds, my_ds]
            return None
        
        # Encode last_route into downscale grid coordinates
        path_local = []
        for pt in self.last_route:
            loc = world_to_downscaled_local(pt[0], pt[1])
            if loc is not None:
                path_local.append(loc)
        path_str = " → ".join(f"({x},{y})" for x, y in path_local) if path_local else "None"

        prompt=open("agents/meeting_challenge/meeting_prompts/navigation_plan.txt","r").read()
        prompt = prompt.format(map=map_str, route=path_str)
        self.logger.debug(f"navigating_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response = self.generator.generate(prompt, img=None, json_mode=False)
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
            waypoints = response_dict.get("waypoints", [])
            if not isinstance(waypoints, list) or len(waypoints) == 0:
                raise ValueError("Waypoints must be a non-empty list.")
            self.last_nav = waypoints
            self.visualize_navigation_plan(
                downscaled_map=downscaled_map,
                original_waypoints=self.last_route if hasattr(self, 'last_route') and isinstance(self.last_route, list) else [],
                waypoints=self.last_nav if hasattr(self, 'last_nav') and isinstance(self.last_nav, list) else [],
                save_path=f"{self.storage_path}/generated_waypoints/navigation_plan_{self.steps:06d}.png"  # or use f"logs/nav_plan_{self.step}.png"
            )
        except Exception as e:
            self.logger.error(
                f"Error generating navigation plan: {e} with traceback: {traceback.format_exc()}. Response was: {response}"
            )
            self.generate_navigation_plan(max_retry=max_retry - 1)

    def visualize_navigation_plan(self, downscaled_map, original_waypoints, waypoints, save_path=None):
        """
        Visualize the downscaled occupancy map with navigation waypoints.

        Args:
            downscaled_map: 2D numpy array (H, W) with values 1=free, 2=obstacle, 3=unknown, 4=agent
            waypoints: List of [x, y] grid coordinates (in downscaled map space)
            agent_grid_pos: [x, y] position of agent in the same grid (downscaled)
            save_path: If provided, save the image to this path
        """
        h, w = downscaled_map.shape

        # Create RGB canvas
        vis_map = np.zeros((h, w, 3), dtype=np.uint8)

        # Color mapping
        vis_map[downscaled_map == 1] = [100, 100, 100]  # free space - gray
        vis_map[downscaled_map == 2] = [255, 255, 255]  # obstacle - white
        vis_map[downscaled_map == 3] = [0, 0, 0]      # unknown - black
        vis_map[downscaled_map == 4] = [0, 255, 0]      # agent - green

        # Convert to float32 for OpenCV drawing
        vis_map = vis_map.astype(np.uint8)

        # Draw waypoints and connect them
        if waypoints and len(original_waypoints) > 0:
            pts = []
            for wp in waypoints:
                x, y = int(wp[0]), int(wp[1])
                if 0 <= x < w and 0 <= y < h:
                    pts.append((x, y))
                    cv2.circle(vis_map, (x, y), radius=2, color=(255, 255, 0), thickness=-1)  # cyan dot

            # Draw lines connecting waypoints
            for i in range(len(pts) - 1):
                cv2.line(vis_map, pts[i], pts[i+1], color=(255, 255, 0), thickness=1)
        if waypoints and len(waypoints) > 0:
            pts = []
            for wp in waypoints:
                x, y = int(wp[0]), int(wp[1])
                if 0 <= x < w and 0 <= y < h:
                    pts.append((x, y))
                    cv2.circle(vis_map, (x, y), radius=2, color=(255, 0, 0), thickness=-1)  # blue dot

            # Draw lines connecting waypoints
            for i in range(len(pts) - 1):
                cv2.line(vis_map, pts[i], pts[i+1], color=(255, 0, 0), thickness=1)

        # Optionally add a scale indicator
        cv2.putText(vis_map, 'Red: Plan', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Resize for better visibility (scale up 4x)
        vis_map = cv2.resize(vis_map, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)

        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            cv2.imwrite(save_path, vis_map)


    def calc_time(self):
        ret=0.
        for i in range(1, len(self.last_route)):
            ret+=np.linalg.norm(np.array(self.last_route[i][:2])-np.array(self.last_route[i-1][:2]))
        return ret
    
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
            agent_pos_description += f"{agent} is now in {agent_pos_dict[agent]['place'] if agent_pos_dict[agent]['place'] is not None else 'open space'}, with coordinate {agent_pos_dict[agent]['pose']}.\n"
        agent_pos_description.strip("\n")
        prompt = prompt.replace("$AgentPoses$", agent_pos_description)
        prompt = prompt.replace("$Places$", self.get_nearest_places_description(self.get_meeting_target()))
        prompt = prompt.replace("$ConversationHistory$", self.get_conversation_description())
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
            response_type = response_dict['type']
            speech = response_dict['speech']
        except Exception as e:
            self.logger.error(
                f"Error getting meeting place: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_type = None
            speech = None
        return response_type, speech

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

    def goto_place(self, target_place: str, force=False) -> (dict, bool):
        places = self.s_mem.get_places() + ['open space']
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
