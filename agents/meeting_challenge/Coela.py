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


class CoelaMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone)
        self.react_freq = 900 # 15min
        if self.debug:
            self.react_freq = 300 # 5 min for debug
        if self.no_react:
            self.react_freq = 1e8
        self.chat_time_limit = 15 # 15 seconds

        self.chatting_with: str = self.scratch["chatting_with"] if "chatting_with" in self.scratch else None # name
        self.chatting_buffer: list[list[datetime, list, str]] = self.scratch["chatting_buffer"] if "chatting_buffer" in self.scratch else []
        for chat in self.chatting_buffer:
            chat[0] = datetime.strptime(chat[0], "%B %d, %Y, %H:%M:%S")
        self.react_mode = None
        self.react_history = []
        self.last_react_time = None
        self.goal_place = None
        self.sleep_time = 0

    def reset(self, name, pose):
        super().reset(name, pose)

    # def _process_obs(self, obs):

    def _process_obs(self, obs):
        # if new day, generate new hourly schedule
        start = time.time()
        if obs['action_status'] == "FAIL":
            self.logger.info(f"{self.name} failed to execute last action {self.react_history[-1]}.")
            if self.react_mode == "speak":
                self.chatting_buffer.pop() # remove the failed chat

        num_new_objects = self.s_mem.update(obs)
        self.logger.debug(f"Process obs 2: {start}, {time.time()}")
        self.curr_events = []

        # react to new objects
        if not self.no_react and num_new_objects > 0:
            new_objects = self.s_mem.object_builder.get_new_objects()
            curr_objects = self.s_mem.object_builder.get_curr_objects()
            kws = [object.name for object in curr_objects]
            img_path = os.path.join(self.storage_path, 'episodic_memory',
                                    f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
            Image.fromarray(obs['rgb']).save(img_path)
            if "gt_seg_entity_idx_to_info" in obs:
                desc = f"I see {', '.join([object.name for object in curr_objects])}."
            else:
                desc = self.generate_captioning(
                    f"Here's an image including {', '.join([object.name for object in new_objects])}. Describe what you see in one sentence. Start with 'I see'.",
                    img=img_path)
                desc += f" Entities detected: {', '.join([object.name for object in curr_objects])}."
            self.last_react_time = self.curr_time
            self.add_event("observation", self.curr_time, self.pose[:3], obs['current_place'], kws, img_path, desc, None)
        
        
        if self.chatting_with is not None:
            if self.chatting_with[0] == "someone":
                subject = self.s_mem.get_name_from_position(self.chatting_with[1])
                if subject is not None:
                    self.chatting_with[0] = subject
                    for chats in self.chatting_buffer:
                        if chats[1][0] == "someone":
                            chats[1] = self.chatting_with
                else:
                    self.logger.error(f"No subject found for the speech event at {self.chatting_with[1]}.")
                    # Image.fromarray(obs['rgb']).save(os.path.join(self.storage_path, 'episodic_memory', f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png'))

            for event in obs['events']:
                if event["type"] == "speech":
                    if event["position"][:2] == self.pose[:2]: # ignore self speech
                        continue
                    subject = self.s_mem.get_name_from_position(event["position"]) # need to deal with more than 2 people chatting
                    if subject is None:
                        self.logger.warning(f"No subject found for the speech event at {event['position']}.")
                    else:
                        self.logger.info(f"{self.name} hears {subject} at {event['position']} says: {event['content']}")
                        if self.chatting_with[0] == "someone":
                            self.chatting_with = [subject, event["position"]]
                            for chats in self.chatting_buffer:
                                if chats[1][0] == "someone":
                                    chats[1] = self.chatting_with
                        if subject == self.chatting_with[0] or self.chatting_with[1] == event["position"]:
                            self.chatting_buffer.append([self.curr_time, self.chatting_with, event["content"]])
        
        start = time.time()
        # react[also save episodic mem] every react_freq seconds or new objects appear
        if len(obs['events']) > 0:
            for event in obs['events']:
                if event["type"] == "speech":
                    if event["position"][:2] == self.pose[:2]:
                        continue
                    subject = self.s_mem.get_name_from_position(event["position"])
                    event["content"] = f"I heard {subject if subject is not None else 'somebody outside of my view'} at {event['position']} says: {event['content']}"
                    kws = [subject, event['type']]
                else:
                    kws = [event["type"]]

                if obs['rgb'] is not None:
                    img_path = os.path.join(self.storage_path, 'episodic_memory', f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
                    Image.fromarray(obs['rgb']).save(img_path)
                else:
                    img_path = None

                self.add_event(event["type"], self.curr_time, event["position"], obs['current_place'], kws, img_path, event["content"], None)
            self.last_react_time = self.curr_time

        if not self.no_react and (self.last_react_time is None or (self.last_react_time != self.curr_time and (self.curr_time - self.last_react_time).total_seconds() > self.react_freq)):
            if obs['rgb'] is not None:

                # todo: get the keywords
                donot_add = False

                img_path = os.path.join(self.storage_path, 'episodic_memory',
                                        f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
                Image.fromarray(obs['rgb']).save(img_path)
                if "gt_seg_entity_idx_to_info" in obs:
                    desc = f"I see {', '.join([object.name for object in self.s_mem.object_builder.get_curr_objects()])}."
                    kws = [object.name for object in self.s_mem.object_builder.get_curr_objects()]
                    if kws == []:
                        donot_add = True
                else:
                    desc = self.generate_captioning(f"Describe what you see in one sentence. Start with 'I see'.", img=img_path)
                    kws = []
                if not donot_add:
                    self.add_event("observation", self.curr_time, self.pose[:3], obs['current_place'], [], img_path, desc, None)
                    self.last_react_time = self.curr_time

        self.logger.debug(f"Process obs 3: {start}, {time.time()}")
    
    def add_event(self, event_type, event_time, event_position, event_place, event_keywords, event_img, event_description, event_text_ft, event_poignancy=None, event_expiration=None):
        event_id = str(len(self.curr_events))
        this_experience = EventInstance(event_id, event_type, event_time, event_time, event_position, event_place, event_keywords, event_img, event_description, event_text_ft, event_poignancy, event_expiration)
        self.curr_events.append(this_experience)

    def _act(self, obs):
        if self.banned:
            if self.pose[0]>-1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        self.logger.debug(f"Current mode is {self.mode}, while the trigger is {self.discussion_trigger}, mode_time_counter is {self.mode_time_counter}")
        action = None
        start = time.time()

        if self.sleep_time > 0:
            self.sleep_time -= 1
            if self.sleep_time == 0:
                return {'type': 'wake', 'arg1': None} # wake up
            return {'type': 'sleep', 'arg1': None}
        
        action = None
        
        if self.goal_place is not None:
            action = self.commute(self.goal_place)
            if action is not None and action['type'] == 'enter' and action['arg1'] == self.goal_place:
                self.sleep_time = 30 * 60
                self.goal_place = None
            if action is not None:
                return action
        
        utterance = None
        if self.chatting_with is not None:
            utterance = self.generate_utterance()
            action = self.conversation(self.chatting_with, utterance)
            self.logger.debug(f"Generate conversation action time: {time.time() - start}")
            if action is not None:
                return action
            action = {'type': 'wait', 'arg1': None}
            utterance = None # no conv
        elif not self.no_react and self.last_react_time == self.curr_time:
            utterance = self.generate_utterance()
            if utterance is None:
                self.logger.warning(f"Failed to generate utterance.")
                return {"type": "wait", "arg1": None}

        # react to the curr_events related retrieved events
        if not self.no_react and self.last_react_time == self.curr_time:
            self.react_mode, react_target = self.generate_react_mode(self.curr_events, utterance)
            self.goal_place = None
            
            if self.react_mode == "speak":
                self.chatting_buffer = []
                self.chatting_with = None
                if utterance == "null":
                    self.logger.info(f"{self.name} stops the conversation.")
                    self.react_mode = "wait"
                    return {"type": "wait", "arg1": None}
                return self.conversation("someone", utterance)
            elif self.react_mode == "go":
                return self.commute(react_target)
            elif self.react_mode == "wait":
                return {
                    'type': 'wait',
                    'arg1': None
                }
            else:
                self.logger.warning(f"Unknown react mode {self.react_mode}.")
                return None

        self.last_action=action
        return self.last_action

    def end_conversation(self):
        self.logger.info(f"{self.name} ends the conversation with {self.chatting_with}.")
        self.chatting_with = None
        self.chatting_buffer = []

    def conversation(self, target: str, content: str):
        WAIT = {'type': 'wait', 'arg1': None}
        if len(self.chatting_buffer) == 0 and (self.chatting_with is None or target is None): # set up the conversation
            curr_events = self.curr_events
            curr_event = curr_events[-1] if len(curr_events) > 0 else None
            for event in curr_events:
                if event.event_type == "speech":
                    curr_event = event
                    break
            if curr_event is not None and curr_event.event_type == "speech":  # response to a conversation
                if target is None or target != curr_event.event_keywords[0]:
                    target = "someone"
                self.chatting_with = target
                self.chatting_buffer.append(
                    [self.curr_time, self.chatting_with, curr_event.event_description.split("] says: ")[1]])
            else:
                self.chatting_with = target
                self.chatting_buffer = []
        
        assert target == self.chatting_with, f"Target {target} is not equal to chatting_with {self.chatting_with}."

        curr_event = None
        for event in self.curr_events:
            if event.event_type == "speech":
                curr_event = event
                break
        if curr_event is not None and curr_event.event_type == "speech":  # response to a conversation
            self.chatting_with = target
            self.chatting_buffer.append(
                [self.curr_time, self.chatting_with, curr_event.event_description.split("] says: ")[1]])
        
        self.logger.info(f"Chatting buffer length: {len(self.chatting_buffer)}")
        if len(self.chatting_buffer) > self.chat_time_limit:
            self.logger.info(f"Chatting with {self.chatting_with} for more than {self.chat_time_limit} seconds. Stop chatting.")
            self.end_conversation()
            return None

        if len(self.chatting_buffer) > 0 and self.chatting_buffer[-1][1] == self.name:
            if (self.curr_time - self.chatting_buffer[-1][0]).total_seconds() > 2:
                self.logger.info(f"{self.chatting_with} is not responding for more than 2 seconds. Stop chatting.")
                self.end_conversation()
                return None
            return WAIT
        
        if content == "null":
            self.logger.info(f"I want to stop the chatting.")
            self.end_conversation()
            return None
        
        self.chatting_buffer.append([self.curr_time + timedelta(seconds=1), (self.name, self.pose[:3]), content])
    
        return {
            'type': 'converse',
            'arg1': content,
            'arg2': 9
        }

    def generate_captioning(self, prompt, img):
        if self.no_react:
            return "Do not revoke the llm in no react mode."
        response = self.generator.generate(prompt, img=img, json_mode=False)
        return response
    
    def delete_quotations(self, text):
        if isinstance(text, str):
            if text.startswith("\"") and text.endswith("\""):
                return text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                return text[1:-1]
        return text

    def generate_utterance(self):
        prompt = open('agents/meeting_challenge/meeting_prompts/coela_prompts/prompt_utterance.txt', 'r').read()
        task_description = open('agents/meeting_challenge/meeting_prompts/task_description.txt', 'r').read()
        prompt = prompt.replace("$TaskDescription$", task_description)
        prompt = prompt.replace("$Character$", self.get_character_description())

        prompt = prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
        prompt = prompt.replace("$Place$", self.current_place if self.current_place is not None else "open space")
        conversation_history_desp = '\n'.join([f"{chat[1][0]}: {chat[2]}" for chat in self.chatting_buffer[-4:]])
        prompt = prompt.replace("$Conversation_history$", conversation_history_desp)
        prompt = prompt.replace("$Context$", self.describe_events(self.curr_events))
        self.logger.debug(f"Utterance prompt: {prompt}")
        response = self.delete_quotations(self.generator.generate(prompt, img=None, json_mode=False))
        self.logger.debug(f"Generated utterance: {response}")
        return response
    
    def generate_react_mode(self, curr_events, utterance):
        if utterance is None:
            prompt = open('agents/meeting_challenge/meeting_prompts/coela_prompts/prompt_react_wo_chat.txt', 'r').read()
        else:
            prompt = open('agents/meeting_challenge/meeting_prompts/coela_prompts/prompt_react.txt', 'r').read()
            prompt = prompt.replace("$Utterance$", utterance)
        task_description = open('agents/meeting_challenge/meeting_prompts/task_description.txt', 'r').read()
        prompt = prompt.replace("$TaskDescription$", task_description)

        prompt = prompt.replace("$Character$", self.get_character_description())
        prompt = prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
        prompt = prompt.replace("$Place$", "None" if self.current_place is None else self.current_place)
        prompt = prompt.replace("$Context$", self.describe_events(curr_events))
        prompt = prompt.replace("$KnownPlaces$", self.get_places_description())
        prompt = prompt.replace("$ActionHistory$", str(self.react_history[-10:]))
        self.logger.debug(f"React prompt: {prompt}")
        response = self.delete_quotations(self.generator.generate(prompt, img=None, json_mode=False))
        self.logger.debug(f"Generated react: {response}")
        if response is None:
            return "wait", None
        if utterance is not None and response.startswith("speak"):
            self.react_history.append("speak")
            return "speak", None
        self.react_history.append(response)
        if response.startswith("go to"):
            return "go", response.split("go to ")[1]
        return "wait", None

    def describe_events(self, events):
        if events is None:
            return "No events."
        desc = ""
        for event in events:
            desc += f"type: {event.event_type}\ntime: {event.event_time}\nplace: {round_numericals(event.event_place)}\nkeywords: {event.event_keywords}\ncontent: {event.event_description}\n\n"
        return desc

    def get_character_description(self):
        """EXAMPLE OUTPUT
           Name: Dolores Heitmiller
           Age: 28
           Innate traits: hard-edged, independent, loyal
           Learned traits: Dolores is a painter who wants live quietly and paint
             while enjoying her everyday life.
           Currently: Dolores is preparing for her first solo show. She mostly
             works from home.
           Lifestyle: Dolores goes to bed around 11pm, sleeps for 7 hours, eats
             dinner around 6pm.
            Groups:
           Daily plan requirement:
           Current Date: Monday, January 1
        """
        return f"""Name: {self.name}
Age: {self.scratch['age']}
Innate traits: {self.scratch['innate']}
Learned traits: {self.scratch['learned']}
Currently: {self.scratch['currently']}
Lifestyle: {self.scratch['lifestyle']}
Groups: {self.scratch['groups']}
Daily plan requirement: {self.scratch['daily_requirement']}
Held objects: {self.held_objects}
Cash: {self.obs['cash']}
Current date: {self.get_curr_date()}
"""