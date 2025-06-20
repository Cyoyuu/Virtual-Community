import numpy as np
import time
import random
import os
from .agent import Agent
from .scene_graph import SceneGraph
from ViCo.tools.utils import is_near_goal

class TourAgent(Agent):
	def __init__(self, name, pose, info, sim_path, tour_spatial_memory, no_react=False, debug=False, logger=None):
		super().__init__(name, pose, info, sim_path, no_react, debug, logger)
		self.detect_interval = 1
		self.s_mem = SceneGraph(os.path.join(self.storage_path, "scene_graph"), detect_interval=self.detect_interval, debug=self.debug, logger=self.logger)
		self.tour_spatial_memory = tour_spatial_memory
		self.tour_building_names = [building for building in tour_spatial_memory.keys() if tour_spatial_memory[building]["bounding_box"] is not None]
		random.shuffle(self.tour_building_names)
		self.curr_idx= 0
		self.curr_goal_name = self.tour_building_names[self.curr_idx]
		curr_goal_dict = self.tour_spatial_memory[self.curr_goal_name]
		self.curr_goal_pos = np.array([curr_goal_dict['places'][0]['location'][0], curr_goal_dict['places'][0]['location'][1]])
		self.curr_goal_bbox = curr_goal_dict["bounding_box"]

	def reset(self, name, pose):
		super().reset(name, pose)
		self.s_mem = SceneGraph(os.path.join(self.storage_path, "scene_graph"), detect_interval=self.detect_interval, debug=self.debug, logger=self.logger) #todo: log the num_frames
		self.curr_idx= 0
		self.curr_goal_name = self.tour_building_names[self.curr_idx]
		curr_goal_dict = self.tour_spatial_memory[self.curr_goal_name]
		self.curr_goal_pos = np.array([curr_goal_dict['places'][0]['location'][0], curr_goal_dict['places'][0]['location'][1]])
		self.curr_goal_bbox = curr_goal_dict["bounding_box"]

	def _process_obs(self, obs):
		self.s_mem.update(obs)
		if self.debug:
			self.s_mem.get_sg().volume_grid_builder.get_occ_map(self.pose[:3], os.path.join(self.storage_path, "semantic_memory", f"occ_map_{self.s_mem.num_frames}.png"))
		self.obs = obs

	def _act(self, obs):
		if obs['current_place'] is not None:
			return {
				'type': 'enter',
				'arg1': 'open space'
			}
		if is_near_goal(self.pose[0], self.pose[1], self.curr_goal_bbox, self.curr_goal_pos):
			self.curr_idx += 1
			if self.curr_idx == len(self.tour_building_names):
				exit()
			self.curr_goal_name = self.tour_building_names[self.curr_idx]
			curr_goal_dict = self.tour_spatial_memory[self.curr_goal_name]
			self.curr_goal_pos = np.array([curr_goal_dict['places'][0]['location'][0], curr_goal_dict['places'][0]['location'][1]])
			self.curr_goal_bbox = curr_goal_dict["bounding_box"]
		self.logger.info(f"{self.name} is navigating to {self.curr_goal_name} at {self.curr_goal_pos}.")
		start = time.time()
		obs["goal_building"] = self.tour_building_names[self.curr_idx]
		obs["goal_pos_xy"] = [self.tour_spatial_memory[self.curr_goal_name]['places'][0]['location'][0], self.tour_spatial_memory[self.curr_goal_name]['places'][0]['location'][1]]
		nav_action = self.navigate(self.s_mem.get_sg(), self.curr_goal_pos, goal_bbox=self.curr_goal_bbox)
		self.logger.info(f"Navigation used {time.time() - start}s.")
		return nav_action
