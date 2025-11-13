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
from ViCo.modules.Amap import Route, RouteNode
from ViCo.tools.utils import *
from ViCo.tools.model_manager import global_model_manager
from agents.sg.builder.builder import Builder, BuilderConfig





class Reasoner(ThinkingModule):
    def __init__(self, generator, logger, name):
        super().__init__(generator, logger, name)
        self.scanned_map = np.zeros(dtype=int, shape=[10, 10])
        self.hsg = HSG()

    def plan(self, curr_time, name, pose, intent):
        prompt = open(f"agents/meeting_challenge/meeting_prompts/query_action.txt", "r").read()
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", name)
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$Intent$", intent)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error extracting ETAs: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict
    
    def overlook(self, bbox, aerial_view, metadata):
        map_index = (int(bbox[0] / 100 + 5), int(bbox[1] / 100 + 5))
        if self.scanned_map[map_index[0], map_index[1]] == 1:
            self.logger.info("This map section already scanned; skipping reconstruction.")
            return None

        self.scanned_map[map_index[0], map_index[1]] = 1
        subgraph = reconstruct_subgraph(aerial_img=aerial_view, metadata=metadata, generator=self.generator)
        self.hsg = merge_subgraph(self.hsg, subgraph)
        self.logger.info(f"Integrated new subgraph into global HSG (total nodes: {len(self.hsg.nodes)})")

    def parse(self, conversation_content):
        prompt = open("agents/meeting_challenge/meeting_prompts/spatial_reasoning_module/hsg/reconstruct_scene_graph.txt", "r").read()
        prompt = prompt.replace("$ConversationExcerpt$", conversation_content)
        self.logger.debug(f"[Parse Prompt]\n{prompt}")

        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            subgraph_json = self.parse_json(prompt, response)
            sub_hsg = HSG()
            for node_data in subgraph_json["nodes"]:
                node = Node(
                    node_id=node_data["node_id"],
                    name=node_data["name"],
                    node_type=node_data["type"],
                    connectivity_center=node_data["connectivity_center"],
                    properties=node_data["properties"],
                    children=node_data["children"],
                )
                sub_hsg.add_node(node)
            for edge in subgraph_json["edges"]:
                sub_hsg.add_edge(edge["from"], edge["to"], edge["relation"], edge["confidence"])
            self.hsg = merge_subgraph(self.hsg, sub_hsg)
            self.logger.info("Updated HSG with new subgraph from conversation.")
        except Exception as e:
            self.logger.error(f"Error parsing HSG from conversation: {e}\n{traceback.format_exc()}")

    def find_safe_path(self, current_node, destination_node):
        """
        Use the current scene graph to find a safe path to the destination.
        If information is missing, identify what needs to be queried.
        """
        prompt_template = open(
            "agents/meeting_challenge/meeting_prompts/spatial_reasoning_module/hsg/generate_safe_path.txt",
            "r"
        ).read()

        known_graph_json = json.dumps(self.hsg.to_graph(), indent=2)
        prompt = (
            prompt_template
            .replace("$StartNode$", current_node)
            .replace("$GoalNode$", destination_node)
            .replace("$KnownGraphJSON$", known_graph_json)
        )

        self.logger.debug(f"pathfinding_prompt: {prompt}")

        response = self.generator.generate(prompt, img=None, json_mode=False)

        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"pathfinding_result: {response_dict}")
        except Exception as e:
            self.logger.error(f"Error parsing pathfinding response: {e}\nResponse was: {response}")
            return None

        # Handle the two cases
        if response_dict.get("status") == "success":
            self.logger.info(f"Safe path found: {response_dict['path']}")
            return response_dict

        elif response_dict.get("status") == "incomplete":
            self.logger.warning("⚠️ Pathfinding incomplete — missing info detected.")
            missing = response_dict.get("missing_info", [])
            actions = response_dict.get("recommended_areas_to_query", [])
            self.logger.info(f"Missing info: {missing}")
            self.logger.info(f"Recommended areas to query: {actions}")

            return response_dict

        else:
            self.logger.error("❌ Unknown status in pathfinding result.")
            return None
        
    def check_waypoint_validity(self, known_sentinel_poses, last_route):
        for wp in last_route:
            for sentinel in known_sentinel_poses:
                if np.linalg.norm(np.array(wp.location[:2])-np.array(sentinel[:2]))<=20:
                    return False
        return True
        
    def refine_waypoints_with_image(self, pose, image, last_route):
        prompt = open(f"agents/meeting_challenge/meeting_prompts/refine_waypoints_aerial_view.txt", "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        self.logger.debug(f"refining waypoints with image {np.array(image).shape}, the original route is {last_route.to_dict()}")
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$DestinationPose$", str(list(last_route[-1].location[:2])))
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=image, json_mode=False)
        try:
            response_dict = self.parse_json_with_image(prompt, image, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error extracting ETAs: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict

    def parse_json_with_image(self, prompt, image, response, last_call=False):
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
                    f"The output format is wrong. Output the formatted json string enclosed in ```json``` only! Do not include any other character in the output!", img=image,
                    chat_history=chat_history)
                return self.parse_json_with_image(None, None, data, last_call=True)
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
        
    def refine_waypoints(self, pose, grid_map, known_sentinel_poses, last_route):
        prompt = open(f"agents/meeting_challenge/meeting_prompts/refine_waypoints.txt", "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        self.logger.debug(f"refining waypoints, the original route is {last_route.to_dict()}")
        grid_map = deepcopy(grid_map)
        def align(x):
            return int(x//20-(-490)//20)
        for sentinel in known_sentinel_poses:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    grid_map[align(sentinel[1]+dy*20)][align(sentinel[0]+dx*20)] = 'D' # sentinel will not appear on the edge
        target_pos = last_route[-1].location
        for wp in last_route:
            y, x = align(wp.location[1]), align(wp.location[0])
            if grid_map[y][x]=='D':
                grid_map[y][x] = 'W'
            else:
                grid_map[y][x] = 'R'
        grid_map[align(target_pos[1])][align(target_pos[0])] = 'T'
        grid_map[align(pose[1])][align(pose[0])] = 'A'
        def map_from_grid(grid):
            map_str_lines = []
            for row in grid:
                line = ''.join(val for val in row)
                map_str_lines.append(line)
            map_str = '\n'.join(map_str_lines)
            return map_str
        prompt = prompt.replace("$Map$", map_from_grid(grid_map))
        self.logger.debug(f"planning_prompt: {prompt}")
        try:
            response = self.generator.generate(prompt, img=None, json_mode=False)
            self.logger.debug(f"generated response: \n{response}\n")
            route = self.parse_route_output(prompt, response)
        except Exception as e:
            self.logger.error(
                f"Error generating query action: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            route = None
        return route
    
    def parse_route_output(self, prompt, response, last_call=False):
        """
        Parse the model's response into (map_str, route_list).
        The model output should contain a map and a JSON list of waypoints.
        """

        try:
            # Extract the map (everything before the JSON)
            map_part = re.split(r'```json', response, maxsplit=1)[0].strip()
            map_lines = [line.strip() for line in map_part.splitlines() if line.strip()]
            # Extract JSON between triple backticks
            json_match = re.findall(r"```json(.*?)```", response, re.DOTALL)
            assert json_match
            # Parse the JSON waypoints
            route_str = json_match[-1].strip()
            route = json.loads(route_str)
            flag = True
            for wp in route:
                if not (0<=wp[0]<50) or not (0<=wp[1]<50) or map_part[wp[0]][wp[1]] not in ['A', 'P', 'T']:
                    flag = False
            assert flag
        except Exception:
                self.logger.warning(f"Error parsing route output, the string was {response}")
                if not last_call:
                    chat_history = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    if flag:
                        data = self.generator.generate(
                            f"The route you generated don't match your map! Make sure they match exactly!",
                            chat_history=chat_history)
                    else:
                        data = self.generator.generate(
                            f"The output format is wrong. Output the formatted map firstly and json string enclosed in ```json``` secondly only! Do not include any other character in the output!",
                            chat_history=chat_history)
                    return self.parse_route_output(None, data, last_call=True)
                else:
                    self.logger.error(f"Error parsing JSON, already last call, the string was {response}")
                    return None

        return route


class SentinelMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False, refine_retry=10):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone)
        self.spatial_resoner = Reasoner(generator=self.generator, logger=self.logger, name=self.name)
        self.emergency = 0
        self.emergency_avoid_target = None
        self.emergency_analysis = {}
        self.ready_to_refine = False
        self.refine_retry = refine_retry
        self.navigation_plan = None

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
                    emergency = 11
                if event['type'] == 'app message':
                    if event['subject'] != self.name: continue
                    if self.last_action['type']=="query_app" and self.last_action['arg1'] == 'query_grid_map_image':
                        image = Image.fromarray(np.array(event['content']).astype(np.uint8))
                        image.save(os.path.join(self.storage_path, f"grid_map_aerial_view_{self.obs['steps']}.png"))
                        route = self.spatial_resoner.refine_waypoints_with_image(self.get_outdoor_pose_description(), image, self.last_route)
                        if route is not None:
                            self.last_route = Route()
                            self.navigation_plan = route
                        else:
                            self.logger.warning(f"Fail to generate new route, still using the original one!")
                    if self.last_action['type']=="query_app" and self.last_action['arg1'] == 'query_refine_route':
                        self.navigation_plan = None
                        if event['content'] is None:
                            time_to_arrival = timedelta(hours=23, minutes=59, seconds=59)
                        else:
                            time_to_arrival = timedelta(seconds=int(event['content'].calc_time(pose=self.get_outdoor_pose())))
                        self.last_route=event["content"]
                        self.last_estimated_arrival_time = self.curr_time + time_to_arrival
                        self.app_message_history.append(Message(self.curr_time, event["subject"], f"The estimated time from current pose to {self.last_action['arg2']} is {time_to_arrival}s"))
                        self.update_known_eta(
                            {
                                self.last_action['arg2']:
                                {
                                    self.name: str(time_to_arrival)
                                }
                            })
        if self.emergency == 0:
            self.emergency = emergency
            if emergency > 0 and not self.last_route.empty():
                self.emergency_analysis["wp_count"] = len(self.last_route)
                self.emergency_analysis["wp_dis"] = np.linalg.norm(np.array(self.pose[:2]) - np.array(self.last_route[0].location[:2]))
        elif 1 <= self.emergency <=10: # if in emergency
            if emergency == 1: # if see sentinel
                self.emergency = emergency # restart emergency
            else: # if no sentinel seen
                self.emergency += 1 # progress the emergency
        else: # if after emergency
            if emergency == 1: # if see emergency
                self.emergency = 1 # restart emergency
            else: # if no sentinel seen
                self.emergency = (self.emergency + 1)%14 # progress the post-emergency
                if not self.last_route.empty():
                    self.logger.debug(f"analyzing emergency, {self.emergency_analysis['wp_count']} and {len(self.last_route)}; {self.emergency_analysis['wp_dis']} and {np.linalg.norm(np.array(self.pose[:2]) - np.array(self.last_route[0].location[:2]))}")
                if not self.last_route.empty() and (self.emergency_analysis["wp_count"] == len(self.last_route) and np.linalg.norm(np.array(self.pose[:2]) - np.array(self.last_route[0].location[:2])) > self.emergency_analysis["wp_dis"]):
                    self.ready_to_refine = True
        if self.current_place is None and not self.last_route.empty() and not self.spatial_resoner.check_waypoint_validity(self.known_sentinel_poses, self.last_route):
            self.ready_to_refine = True

    def _act(self, obs):
        if self.banned:
            if self.pose[0]>-1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        # if self.grid_map is None:
        #     self.last_action = {"type": "query_app", "arg1": "query_grid_map"}
        #     return self.last_action
        # if still in emergency
        if 1 <= self.emergency <= 10:
            if self.emergency_avoid_target is None or is_near_goal(self.pose[0], self.pose[1], None, list(self.emergency_avoid_target)):
                self.emergency_avoid_target = self.emergency_avoid()
            if self.emergency_avoid_target is None:
                self.logger.warning(f"I cannot find a suitable avoidance!")
                self.last_action = None
                return None
            else:
                self.logger.info(f"performing emergency avoiding. Target is {self.emergency_avoid_target}")
                self.last_action = self.navigate(self.s_mem.get_sg(), list(self.emergency_avoid_target))
                return self.last_action
        elif self.emergency > 10:
            self.emergency_avoid_target = None
            self.logger.info(f"after emergency avoiding. emergency level is {self.emergency}")
            self.last_action = {'type': 'turn_right', 'arg1': 90}
            return self.last_action
        else:
            self.emergency_avoid_target = None
            if self.ready_to_refine and self.refine_retry > 0:
                self.refine_retry -= 1
                self.ready_to_refine = False
                action = {"type": "query_app", "arg1": "query_grid_map_image", "arg2": [pose[:2] for pose in self.known_sentinel_poses], "arg3": [wp.location for wp in self.last_route]}
                self.last_action = action
                return self.last_action
                # route = self.spatial_resoner.refine_waypoints(self.pose, self.grid_map, self.known_sentinel_poses, self.last_route)
                # if route is not None:
                #     self.last_route = Route()
                #     for wp in route:
                #         wp[0], wp[1] = (wp[1] + (-490)//20)*20+10, (wp[0] + (-490)//20)*20+10
                #         self.last_route.append(RouteNode(list(wp), 'walk', datetime.combine(self.curr_time.date(), datetime.strptime("23:59:59", "%H:%M:%S").time())))
                # else:
                #     self.logger.warning(f"Fail to generate new route, still using the original one!")
            if self.navigation_plan is not None:
                action = {"type": "query_app", "arg1": "query_refine_route", "arg2": self.navigation_plan}
                self.last_action = action
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
                action = self.discuss_act()
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
            if np.linalg.norm(np.array(self.pose[:2])-np.array(sentinel_pose[:2])) < 40:
                near_sentinels.append(sentinel_pose)
        self.logger.info(f"calculating emergency avoidance, near sentinels include {near_sentinels}")
        # get occ map
        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder
        occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map() # occ map: 1 for unknow, 2 for obstacle, 3 for open
        def valid(x, y):
            px, py = builder.align_nav(x)-x_min, builder.align_nav(y)-y_min
            return 0 <= int(py) < y_max - y_min and 0 <= int(px) < x_max - x_min and occ_map[int(py)][int(px)] not in [2, 4]
        valid_pos = []
        def calc(x, y):
            ret = 0
            for sentinel_pose in near_sentinels:
                ret += (x - self.pose[0])/(sentinel_pose[0]-self.pose[0]) + (y - self.pose[1])/(sentinel_pose[1]-self.pose[1])
            return ret
        minv, maxp = 0, None
        for wp in self.last_nav:
            if not valid(wp[0], wp[1]): continue
            value = calc(wp[0], wp[1])
            if value < minv:
                minv = value
                maxp = wp
        if maxp is not None:
            self.logger.debug(f"reasonable wp exists in self.last_nav, just use that")
            return maxp
        for x in range(int(self.pose[0])-40, int(self.pose[0])+40):
            for y in range(int(self.pose[1])-40, int(self.pose[1])+40):
                if not valid(x, y): continue
                value = calc(x, y)
                if value < 0:
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
