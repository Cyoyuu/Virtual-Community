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
from modules.Amap import Route, RouteNode
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
        
    def check_waypoint_validity(self, known_sentinel_poses, last_route, excluding_wps=[]):
        for wp in last_route:
            if any(np.linalg.norm(np.array(wp.location[:2])-np.array(excluding_wp[:2]))<5 for excluding_wp in excluding_wps): continue
            for sentinel in known_sentinel_poses:
                if np.linalg.norm(np.array(wp.location[:2])-np.array(sentinel[:2]))<=20:
                    return False, str(sentinel[:2])
        return True, "None"
        
    def refine_waypoints_with_image(self, pose, image, reference_route, known_sentinel_poses, danger):
        prompt = open(f"agents/meeting_challenge/meeting_prompts/refine_waypoints_aerial_view.txt", "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        self.logger.debug(f"refining waypoints with image {np.array(image).shape}, the original route is {reference_route}")
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$DestinationPose$", str(list(reference_route[-1][:2])))
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=image, json_mode=False)
        prompt = prompt.replace("$KnownSentinelPoses$", known_sentinel_poses)
        prompt = prompt.replace("$Danger$", danger)
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
        
class CoSaRDiscusser(Discusser):
    def __init__(self, generator, logger, name, type="", ablate=""):
        super().__init__(generator, logger, name, type, ablate)

    def start(self, agent_names, agent_opinions, places, conversation_history):
        prompt = open(os.path.join(self.prompt_path, "decide_start.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$AgentList$", agent_names)
        prompt = prompt.replace("$AgentOpinions$", agent_opinions)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
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
    
    def conclude_and_decide_and_extract(self, curr_time, agent_names, agent_opinions, places, conversation_history):
        prompt = open(os.path.join(self.prompt_path, "conclude_and_decide_and_extract.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$AgentList$", agent_names)
        prompt = prompt.replace("$AgentOpinions$", agent_opinions)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error concluding opinions: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict
    
    def analyze_and_plan(self, curr_time, pose, agent_opinions, places, conversation_history, known_poses, known_eta, known_sentinel_poses, stalling, speech):
        prompt_path = self.prompt_path
        if "spatial_memory" in self.ablate: prompt_path = os.path.join(prompt_path, "no_spatial_memory")
        prompt = open(os.path.join(prompt_path, "analyze_and_plan.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$AgentOpinions$", agent_opinions)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        prompt = prompt.replace("$KnownPoses$", known_poses)
        prompt = prompt.replace("$KnownETA$", known_eta)
        prompt = prompt.replace("$KnownSentinelPoses$", known_sentinel_poses)
        prompt = prompt.replace("$Speech$", speech)
        if stalling:
            prompt = prompt.replace("$Stalling$", "The discussion has extended too long. Avoid throwing new questions and finalize as soon as possible!")
        else:
            prompt = prompt.replace("$Stalling$", '')
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
    
    def speak(self, curr_time, pose, agent_opinions, places, conversation_history, known_poses, known_eta, known_sentinel_poses, missing_info, stalling):
        prompt_path = self.prompt_path
        if "spatial_memory" in self.ablate: prompt_path = os.path.join(prompt_path, "no_spatial_memory")
        prompt = open(os.path.join(prompt_path, "speak_speak.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$AgentOpinions$", agent_opinions)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        prompt = prompt.replace("$KnownPoses$", known_poses)
        prompt = prompt.replace("$KnownETA$", known_eta)
        prompt = prompt.replace("$KnownSentinelPoses$", known_sentinel_poses)
        prompt = prompt.replace("$MissingInfo$", missing_info)
        if stalling:
            prompt = prompt.replace("$Stalling$", "The discussion has extended too long. Avoid throwing new questions and finalize as soon as possible!")
        else:
            prompt = prompt.replace("$Stalling$", '')
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
        try:
            response = self.generator.generate(prompt, img=None, json_mode=False)
            self.logger.debug(f"generated response: {response}")
        except Exception as e:
            self.logger.error(
                f"Error extracting ETAs: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response = None
        return response



class CoSaRMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False, refine_retry=5, ablate="", server_port=8000):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone, ablate=ablate, server_port=server_port)
        self.logger.info(f"client port: {self.generator._mm_client.server_port}")
        self.decider = Decider(generator=self.generator, logger=self.logger, name=self.name, type='cosar', ablate=self.ablate)
        self.discusser = CoSaRDiscusser(generator=self.generator, logger=self.logger, name=self.name, type='cosar', ablate=self.ablate)
        self.spatial_resoner = Reasoner(generator=self.generator, logger=self.logger, name=self.name)
        # emergency property
        self.emergency = 0
        self.emergency_avoid_target = None
        self.emergency_analysis = {}
        # route refine property
        self.ready_to_refine = False
        self.refine_reference = None
        self.refine_target_place = None # place the reference route lead to
        self.refine_target_danger = None
        self.coarse_refine_result = None
        self.max_refine_retry = refine_retry
        self.max_continuous_refine_retry = 2
        self.continuous_refine_retry = self.max_continuous_refine_retry
        self.refine_retry = dict()
        # CoSaR property
        self.rethink = True
        self.chat_time_limit = 60

        self.refine_history = []

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
        self.process_obs_with_sptial_knowledge(obs, enable_refine=True)
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
                        # route_validity, danger = self.spatial_resoner.check_waypoint_validity(self.known_sentinel_poses, self.last_route, excluding_wps=[self.pose[:2], self.last_route[-1].location])
                        route = self.spatial_resoner.refine_waypoints_with_image(self.get_outdoor_pose_description(), image, self.refine_reference, self.get_known_sentinel_poses_description(), self.refine_target_danger)
                        if route is not None:
                            self.coarse_refine_result = route
                        else:
                            self.logger.warning(f"Fail to generate new route, still using the original one!")
                    if self.last_action['type']=="query_app" and self.last_action['arg1'] == 'query_refine_route':
                        self.logger.debug(f"successfully got refined route from the app. it's {event['content']['refined_route'].to_dict()}")
                        if event['content'] is None:
                            time_to_arrival = timedelta(hours=23, minutes=59, seconds=59)
                        else:
                            time_to_arrival = timedelta(seconds=int(event['content']["refined_route"].calc_time(pose=self.get_outdoor_pose())))
                        if self.goal_place==self.refine_target_place:
                            self.last_route=event["content"]["refined_route"]
                            self.last_estimated_arrival_time = self.curr_time + time_to_arrival
                        image = Image.fromarray(np.array(event['content']["grid_map_image"]).astype(np.uint8))
                        image.save(os.path.join(self.storage_path, f"grid_map_aerial_view_with_refined_route_{self.obs['steps']}.png"))
                        self.logger.debug(f"successfully saved refined route image to grid_map_aerial_view_with_refined_route_{self.obs['steps']}.png")
                        route_validity, danger = self.spatial_resoner.check_waypoint_validity(self.known_sentinel_poses, event["content"]["refined_route"], excluding_wps=[self.pose[:2], event["content"]["refined_route"][-1].location])
                        self.refine_history.append(route_validity)
                        json.dump(self.refine_history, open(os.path.join(self.storage_path, "refine_history.json"), "w"))
                        if route_validity or self.refine_retry[self.refine_target_place] < 0 or self.continuous_refine_retry < 0:
                            self.logger.info(f"successful route refinement or reach refinement retrying limit.")
                            self.ready_to_refine = False
                            self.coarse_refine_result = None
                            self.continuous_refine_retry = self.max_continuous_refine_retry
                            self.update_known_eta(
                                {
                                    self.refine_target_place:
                                    {
                                        self.name: str(time_to_arrival)
                                    }
                                })
                        else:
                            self.logger.info(f"unsuccessful route refinement, do it again.")
                            self.ready_to_refine = True
                            self.refine_target_danger = danger
                            self.refine_reference = self.refine_reference
                            self.coarse_refine_result = None
                            self.continuous_refine_retry -= 1
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
                # if not self.last_route.empty() and (self.emergency_analysis["wp_count"] == len(self.last_route) and np.linalg.norm(np.array(self.pose[:2]) - np.array(self.last_route[0].location[:2])) > self.emergency_analysis["wp_dis"]): # did not pass in emergency
                #     self.ready_to_refine = True
                #     self.refine_reference = [list(wp.location) for wp in self.last_route]
        if self.mode_time_counter % 60 == 0: self.continuous_refine_retry = self.max_continuous_refine_retry
        if self.last_route and not self.ready_to_refine:
            route_validity, danger = self.spatial_resoner.check_waypoint_validity(self.known_sentinel_poses, self.last_route, excluding_wps=[self.pose[:2], self.last_route[-1].location])
            if self.current_place is None and (not route_validity or self.emergency > 10) and self.coarse_refine_result is None:
                self.refine_target_place = self.goal_place
                if self.refine_target_place not in self.refine_retry:
                    self.refine_retry[self.refine_target_place] = self.max_refine_retry
                if self.refine_retry[self.refine_target_place] >= 0 and self.continuous_refine_retry >= 0:
                    self.ready_to_refine = True
                    self.refine_target_danger = danger
                    self.refine_reference = [list(wp.location) for wp in self.last_route]
                else:
                    self.ready_to_refine = False
            else:
                self.ready_to_refine = False

    def _act(self, obs):
        if self.banned:
            if self.pose[0]>-1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        if self.max_refine_retry<=0 or (self.goal_place is not None and self.goal_place in self.refine_retry and self.refine_retry[self.goal_place] < 0) or self.continuous_refine_retry < 0:
            self.emergency = 0
        if 1 <= self.emergency <= 10: # if still in emergency
            self.logger.info(f"In emergency, emergency level is {self.emergency}")
            # if self.current_place is not None:
            #     self.emergency = max(9, self.emergency)
            #     self.last_action={"type": "wait"}
            #     return self.last_action
            # if len(self.obs['accessible_places']) > 0 :
            #     emergency_avoid_target_place = self.goal_place if self.goal_place in self.obs['accessible_places'] else self.obs['accessible_places'][0]
            #     self.logger.info(f"performing emergency avoiding. Target is {emergency_avoid_target_place}")
            #     self.last_action = {
            #         'type': 'enter',
            #         'arg1': emergency_avoid_target_place
            #     }
            #     return self.last_action
            self.rethink = False
            self.city_navigate(self.goal_place, requery=False) # just for refresh self.nav
            self.rethink = True
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
        elif self.emergency > 10: # if after emergency
            self.emergency_avoid_target = None
            self.logger.info(f"after emergency avoiding. emergency level is {self.emergency}")
            self.last_action = {'type': 'turn_right', 'arg1': 90}
            return self.last_action
        else:  # if not in emergency
            self.emergency_avoid_target = None
        # route refinement
        if self.ready_to_refine:
            if self.refine_retry[self.refine_target_place] > 0:
                self.refine_retry[self.refine_target_place] -= 1
                self.ready_to_refine = False
                action = {"type": "query_app", "arg1": "query_grid_map_image", "arg2": [pose[:2] for pose in self.known_sentinel_poses], "arg3": self.refine_reference}
                self.last_action = action
                return self.last_action
            elif self.refine_retry[self.refine_target_place] == 0: # if 0, make a normal query
                self.refine_retry[self.refine_target_place] -= 1
                self.ready_to_refine = False
                action = {"type": "query_app", "arg1": "query_route", "arg2": self.goal_place}
                self.last_action = action
                return self.last_action
            else:
                self.logger.error(f"Error in refine retry count for place {self.refine_target_place}, the count is {self.refine_retry[self.refine_target_place]}")
                raise ValueError(f"Error in refine retry count for place {self.refine_target_place}, the count is {self.refine_retry[self.refine_target_place]}")
        if self.coarse_refine_result is not None:
            action = {"type": "query_app", "arg1": "query_refine_route", "arg2": [pose[:2] for pose in self.known_sentinel_poses], "arg3": self.refine_reference, "arg4": self.coarse_refine_result}
            self.last_action = action
            return self.last_action
        # no emergency
        if any([sentinel[3]==0 for sentinel in self.known_sentinel_poses]):
            speech = f"I saw sentinel(s) at {[sentinel[:3] for sentinel in self.known_sentinel_poses if sentinel[3]==0]}"
            self.last_action = {"type": "remote_converse", "arg1": speech, "arg2": 3200}
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
                if self.mode_time_counter > self.chat_time_limit:
                    if self.meeting_place is None:
                        self.logger.warning(f"Exceeding discussion limit but no agreed location. Terminating the task.")
                        action = {"type": "task_terminate"}
                        return action
                    else:
                        self.logger.warning(f"Exceeding discussion limit. Going to the most preferred location")
                        self.enter_navigation_mode(goal_place=self.meeting_place)
                        return {"type": "wait"}
                action = self.discuss_act()
            elif self.mode == NavAgentState.NAVIGATE:
                self.mode_time_counter += 1
                action, arrived = self.city_navigate(self.goal_place, requery=False)
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

    def discuss_process_speech(self, obs):
        agent_names = ", ".join(obs["agent_pos_dict"].keys())
        places = self.get_nearest_places_description(self.get_meeting_target())
        current_message = self.get_conversation_description(limit=1)
        conversation_history = self.get_conversation_description()
        curr_time = self.curr_time.strftime('%H:%M:%S')
        extracted_info={}

        for agent_name in obs["agent_pos_dict"].keys():
            if agent_name not in self.agent_opinions:
                self.agent_opinions[agent_name] = 'undecided'

        if self.mode==NavAgentState.NAVIGATE and self.rethink:
            extracted_info = self.discusser.start(agent_names=agent_names, agent_opinions=self.get_agent_opinions_description(), places=places, conversation_history=current_message)
            self.agent_opinions = extracted_info['agent_opinions']
            if "initiate_discussion" in extracted_info and extracted_info["initiate_discussion"]:
                self.enter_discussion_mode(trigger="NEW DISCUSSION")
        elif self.mode==NavAgentState.DISCUSS:
            extracted_info = self.discusser.conclude_and_decide_and_extract(curr_time=curr_time, agent_names=agent_names, agent_opinions=self.get_agent_opinions_description(), places=places, conversation_history=current_message)
            self.agent_opinions = extracted_info['agent_opinions']
            decision = extracted_info['agreement_check']
            if decision["agreed_location"] is not None:
                meeting_place = decision["agreed_location"]
                if meeting_place.startswith("<") and meeting_place.endswith(">"):
                    meeting_place = meeting_place[1:-1]
                self.meeting_place = meeting_place
            if decision["agreement_reached"] == True:
                self.enter_navigation_mode(goal_place=self.meeting_place)
        if 'spatial_memory' not in self.ablate and 'analyzer' not in self.ablate:
            if 'ETA Map' in extracted_info:
                self.update_known_eta(extracted_info['ETA Map'])#!!!
            if 'Agent Poses' in extracted_info:
                self.update_known_poses(extracted_info['Agent Poses'])
            if 'Sentinel Poses' in extracted_info:
                self.update_known_sentinel_poses(extracted_info['Sentinel Poses'], shared=1)
    
    def discuss_act(self):
        action = None
        agent_names = ", ".join(self.obs["agent_pos_dict"].keys())
        places = self.get_nearest_places_description(self.get_meeting_target())
        conversation_history = self.get_conversation_description()
        app_message = self.get_app_message_description()
        curr_time = self.curr_time.strftime('%H:%M:%S')

        # Reasoner
        speech = self.discusser.speak(curr_time=curr_time, pose=self.get_outdoor_pose_description(), agent_opinions=self.get_agent_opinions_description(), places=places, conversation_history=conversation_history, known_poses=self.get_known_poses_description(), known_eta=self.get_known_eta_description(), known_sentinel_poses=self.get_known_sentinel_poses_description(), missing_info="missing_info", stalling=self.mode_time_counter>30)
        self.discussion_plan = self.discusser.analyze_and_plan(curr_time=curr_time, pose=self.get_outdoor_pose_description(), agent_opinions=self.get_agent_opinions_description(), places=places, conversation_history=conversation_history, known_poses=self.get_known_poses_description(), known_eta=self.get_known_eta_description(), known_sentinel_poses=self.get_known_sentinel_poses_description(), stalling=self.mode_time_counter>30, speech=speech['speech'])
        # missing_info="\n".join(self.discussion_plan['missing info']) # not in use now
        action = {"type": "wait"}
        # Executor
        if self.discussion_plan["action"]=="wait":
            action = {"type": "wait"}
            self.discussion_plan = None
        elif self.discussion_plan['action']=='goto':
            if self.discussion_plan['content'].startswith("<") and self.discussion_plan['content'].endswith(">"):
                self.discussion_plan['content'] = self.discussion_plan['content'][1:-1]
            action = self.city_navigate(goal_place=self.discussion_plan['content'])[0]
        elif self.discussion_plan["action"] in ["query_place", "query place"]:
            if self.discussion_plan['content'].startswith("<") and self.discussion_plan['content'].endswith(">"):
                self.discussion_plan['content'] = self.discussion_plan['content'][1:-1]
            action = {'type': 'query_app', 'arg1': 'query_place', 'arg2': self.discussion_plan['content']}
        elif self.discussion_plan['action'] in ['query_route', 'query route']:
            if self.discussion_plan['content'].startswith("<") and self.discussion_plan['content'].endswith(">"):
                self.discussion_plan['content'] = self.discussion_plan['content'][1:-1]
            places_in_mem = self.s_mem.get_places()
            if self.discussion_plan['content'] not in places_in_mem:
                self.logger.warning(f"the place {self.discussion_plan['content']} is not in memory, cannot query route")
                for place_in_mem in places_in_mem:
                    if self.discussion_plan['content'] in place_in_mem:
                        self.logger.warning(f"but found similar place {place_in_mem} in memory, maybe you want to go there?")
                        self.discussion_plan['content'] = place_in_mem
                        break
            action = {'type': 'query_app', 'arg1': 'query_route', 'arg2': self.discussion_plan['content']}
        elif self.discussion_plan["action"]=="speak":
            action = {"type": "remote_converse", "arg1": speech['speech'], "arg2": 3200}
        else:
            raise NotImplementedError(f"discussion plan type {self.discussion_plan['action']} is not supported")
        return action
    
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
    
    def generate_navigation_plan_with_img(self, max_retry=3):
        assert max_retry >= 0
        if max_retry == 0:
            # Fallback: use first 3 waypoints from last_route
            nav_horizon = 0
            self.last_nav = []
            while nav_horizon < len(self.last_route) and nav_horizon < 3 and self.last_route[nav_horizon].transit == 'walk':
                self.last_nav.append(self.last_route[nav_horizon].location)
                nav_horizon += 1
            return
        # get occ map
        builder = self.s_mem.get_sg(place=self.current_place).volume_grid_builder
        occ_map, x_min, y_min, x_max, y_max = builder.get_occ_map() # occ map: 1 for unknown, 2 for obstacle, 3 for open, 4 for warning
        agent_x_world, agent_y_world = self.pose[0], self.pose[1]
        agent_pos_in_map = [
            builder.align_nav(agent_x_world) - x_min,
            builder.align_nav(agent_y_world) - y_min
        ]
        # Define local crop around agent (±30m in world coordinates)
        x_low_w, x_up_w = max(agent_x_world - 30, -400), min(agent_x_world + 30, 400)
        y_low_w, y_up_w = max(agent_y_world - 30, -400), min(agent_y_world + 30, 400)
        # Convert to map indices
        x_low = int(max(0, builder.align_nav(x_low_w) - x_min))
        x_up = int(min(occ_map.shape[1], builder.align_nav(x_up_w) - x_min))
        y_low = int(max(0, builder.align_nav(y_low_w) - y_min))
        y_up = int(min(occ_map.shape[0], builder.align_nav(y_up_w) - y_min))
        def get_pos_in_map(pos):
            return [builder.align_nav(pos[0]) - x_min - x_low, builder.align_nav(pos[1]) - y_min - y_low]

        # Crop the occupancy map
        if x_low >= x_up or y_low >= y_up:
            raise ValueError("Invalid crop bounds after alignment.")

        cropped_map = occ_map[y_low:y_up, x_low:x_up]  # Note: y first, then x

        img = self.visualize_occ_map_with_objects(cropped_map, min_x=x_low_w, max_x=x_up_w, min_y=y_low_w, max_y=y_up_w, route_coords=[wp.location for wp in self.last_route], circle_coords=[sentinel_pos[:2] for sentinel_pos in self.visible_sentinels.values()], agent_coords=[self.pose[:2]], target_coords=[self.last_route[-1].location])
        
        # Encode full route and goal
        path_local = []
        path_global = []
        for pt in self.last_route:
            if x_low_w <= pt.location[0] <= x_up_w and y_low_w <= pt.location[1] <= y_up_w:
                path_local.append(pt.location)
            path_global.append(pt.location)

        route_local_str = " → ".join(f"({x:.1f},{y:.1f})" for x, y in path_local) if path_local else "None"
        route_global_str = " → ".join(f"({x:.1f},{y:.1f})" for x, y in path_global) if path_global else "None"
        goal_grid = path_global[-1] if path_global else "unknown"
        route_global_str += " (goal)"

        prompt=open("agents/meeting_challenge/meeting_prompts/navigation_plan_with_img.txt","r").read()
        # Format the prompt
        prompt = prompt.replace("$TaskDescription$", self.spatial_resoner.task_decription)
        prompt = prompt.replace("$SelfPose$", str(self.pose[:2]))
        prompt = prompt.replace(
            "$route_local$", route_local_str
        )
        prompt = prompt.replace(
            "$route_global$", route_global_str
        )
        prompt = prompt.replace(
            "$goal_grid$", str(goal_grid) if isinstance(goal_grid, list) else "unknown"
        )
        prompt = prompt.replace(
            "$KnownSentinelPoses$", json.dumps(self.visible_sentinels)
        )
        self.logger.debug(f"navigating_prompt: {prompt}")
        try:
            response = self.generator.generate(prompt, img=img, json_mode=False)
            response_dict = self.parse_json_with_image(prompt, img, response)
            self.logger.debug(f"generated response: {response_dict}")
            waypoints = response_dict
            if not isinstance(waypoints, list) or len(waypoints) == 0:
                raise ValueError("Waypoints must be a non-empty list.")
            self.last_nav = waypoints
            if self.debug:
                self.visualize_occ_map_with_objects(
                    cropped_map,
                    min_x=x_low_w, max_x=x_up_w, min_y=y_low_w, max_y=y_up_w, route_coords=[wp.location for wp in self.last_route], circle_coords=[sentinel_pos[:2] for sentinel_pos in self.visible_sentinels.values()], agent_coords=[self.pose[:2]], target_coords=[self.last_route[-1].location], new_route_coords=self.last_nav,
                    save_path=f"{self.storage_path}/generated_waypoints/navigation_plan_{self.steps:06d}.png"
                )
        except Exception as e:
            self.logger.error(
                f"Error generating navigation plan: {e} with traceback: {traceback.format_exc()}. Response was: {response}"
            )
            self.generate_navigation_plan_with_img(max_retry=max_retry - 1)

    def visualize_occ_map_with_objects(self, cropped_map, route_coords=None, circle_coords=None, agent_coords=None, target_coords=None, new_route_coords=None, save_path=None, 
                                resolution=1.0, min_x=0, min_y=0, max_x=0, max_y=0):
        # Build RGB map
        h, w = cropped_map.shape
        draw_map = np.zeros((h, w, 3), dtype=np.uint8)
        self.logger.debug(f"h and w are {h} and {w}, agent_coords is {agent_coords}")

        # Map semantics: 1=unknown(gray), 2=obstacle(black), 3=open(white), 4=warning(red)
        draw_map[cropped_map == 1] = [128, 128, 128]   # gray
        draw_map[cropped_map == 2] = [0, 0, 0]   # black
        draw_map[cropped_map == 3] = [255, 255, 255]         # white
        draw_map[cropped_map == 4] = [255, 0, 0]       # red

        # --- Plot background grid ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(draw_map, origin='lower', extent=[min_x, max_x, min_y, max_y])

        # Helper: test if a coordinate is in free space
        def is_free(x, y):
            gx = int((x) / resolution)
            gy = int((y) / resolution)
            if gx < min_x or gx >= max_x or gy < min_y or gy >= max_y:
                return False
            # cropped_map: 3 = open (free)
            return True

        # --- 2️⃣ Circle-like coordinates ---
        if circle_coords is not None:
            for (x, y) in circle_coords:
                # if is_free(x, y):
                circle = Circle((x, y), radius=5, edgecolor='red',
                                facecolor='red', linewidth=1.5)
                ax.add_patch(circle)

        # --- 1️⃣ Route-like coordinates ---
        if route_coords is not None and len(route_coords) > 0:
            # self.logger.info(f"getting grid image, route_coords are {route_coords}")
            route_coords = np.array([p for p in route_coords if is_free(p[0], p[1])])
            if len(route_coords) > 0:
                ax.plot(route_coords[:, 0], route_coords[:, 1], c='blue', linewidth=1.5)
                ax.scatter(route_coords[:, 0], route_coords[:, 1],
                        s=12, c='blue', edgecolors='k', linewidths=0.3, zorder=3)

        # --- 1️⃣ New Route-like coordinates ---
        if new_route_coords is not None and len(new_route_coords) > 0:
            new_route_coords = np.array([p for p in new_route_coords if is_free(p[0], p[1])])
            if len(new_route_coords) > 0:
                ax.plot(new_route_coords[:, 0], new_route_coords[:, 1], c='orange', linewidth=1.5)
                ax.scatter(new_route_coords[:, 0], new_route_coords[:, 1],
                        s=12, c='orange', edgecolors='k', linewidths=0.3, zorder=3)

        # --- 3️⃣ Point-like coordinates ---
        if agent_coords is not None:
            pts = np.array([p for p in agent_coords])
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=12, c='green', zorder=3)

        # --- 3️⃣ Point-like coordinates ---
        if target_coords is not None:
            pts = np.array([p for p in target_coords if is_free(p[0], p[1])])
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=12, c='purple', zorder=3)

        # --- Display setup ---
        ax.set_xlabel("X (world units)")
        ax.set_ylabel("Y (world units)")
        ax.set_title("Occupancy Grid with External Elements")
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.axis('equal')
        plt.tight_layout(pad=0., rect=[0, 0, 1, 1])

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        h, w, _ = buf.shape
        img = buf[:, :, :3]  # drop alpha
        if img.shape[0]>800:
            img = img[::2, ::2, :]
        img = Image.fromarray(np.array(img).astype(np.uint8))
        if save_path is not None:
            img.save(save_path)
        plt.close(fig)
        return img

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
