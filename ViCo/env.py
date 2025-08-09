import logging
import os
import sys
import time
import shutil, errno
import argparse
import random
from datetime import datetime, timedelta
from collections import defaultdict
import tqdm
import traceback
import multiprocessing as mp

from gymnasium import Env, spaces
import genesis as gs
from genesis.utils.tools import FPSTracker
import genesis.utils.geom as geom_utils
from genesis.options import CoacdOptions
from genesis.engine.entities.rigid_entity import RigidEntity, RigidVisGeom
import numpy as np
import json
import string
from PIL import Image
import random

current_directory = os.getcwd()
sys.path.insert(0, current_directory)

from ViCo.tools.constants import LIGHTS, ENV_OTHER_METADATA
from ViCo.tools.utils import *
from ViCo.modules import *

from agents import Agent, AgentLogger

class VicoEnv:
	def __init__(self,
				 seed,
				 precision,
				 logging_level,
				 backend=gs.cpu,
				 head_less=False,
				 resolution=512,
				 challenge='full',
				 num_agents=5,
				 config_path='',
				 scene='NY',
				 enable_indoor_scene=False,
				 enable_outdoor_objects=False,
				 outdoor_objects_max_num=10,
				 enable_collision=False,
				 skip_avatar_animation=False,
				 enable_gt_segmentation=False,
				 no_load_scene=False,
				 output_dir='output',
				 enable_third_person_cameras=True,
				 no_traffic_manager=False,
				 tm_vehicle_num=0,
				 tm_avatar_num=0,
				 enable_tm_debug=False,
				 save_per_seconds=10,
				 defer_chat=False,
				 debug=False,
				 enable_indoor_objects=False):
		if not gs._initialized:
			gs.init(seed=seed, precision=precision, logging_level=logging_level, backend=backend)
		fh = logging.FileHandler(os.path.join(output_dir, 'raw.log'))
		fh.setLevel(logging.DEBUG)
		gs.logger._logger.addHandler(fh)
		self.resolution = resolution
		self.challenge = challenge
		self.num_agents = num_agents
		self.output_dir = output_dir
		self.enable_third_person_cameras = enable_third_person_cameras
		self.save_per_seconds = save_per_seconds
		self.seed = seed
		self.skip_avatar_animation = skip_avatar_animation
		self.defer_chat = defer_chat
		self.debug = debug
		self.enable_indoor_scene = enable_indoor_scene
		if self.enable_indoor_scene:
			self.coarse_indoor_scene = json.load(open("ViCo/modules/indoor_scenes/coarse_type_to_indoor_scene.json", 'r'))
		self.active_places_info = {}
		self.active_places_agents = defaultdict(list)
		self.enable_outdoor_objects = enable_outdoor_objects
		self.enable_indoor_objects = enable_indoor_objects
		self.outdoor_objects_max_num = outdoor_objects_max_num
		self.enable_collision = enable_collision
		self.enable_gt_segmentation = enable_gt_segmentation
		self.scene_name = scene
		self.entity_idx_to_info = defaultdict(dict)
		self.entity_idx_to_color = []
		self.place_cameras = {}

		self.config_path = config_path
		self.config = json.load(open(os.path.join(self.config_path, 'config.json'), 'r'))
		if "dt_agent" not in self.config:
			self.config["dt_agent"] = 0
		if "dt_sim" not in self.config:
			self.config["dt_sim"] = 0
		if "dt_chat" not in self.config:
			self.config["dt_chat"] = 0

		if self.challenge == 'full':
			assert self.num_agents == self.config[
				'num_agents'], f"num_agents in config file is {self.config['num_agents']}, but got {self.num_agents}"

		self.curr_time: datetime = datetime.strptime(self.config['curr_time'], "%B %d, %Y, %H:%M:%S")
		self.steps = self.config['step']
		self.sec_per_step = self.config['sec_per_step']
		self.seconds = self.steps * self.sec_per_step
		self.building_metadata = json.load(open(os.path.join(config_path, "building_metadata.json"), 'r'))
		self.place_metadata = json.load(open(os.path.join(config_path, "place_metadata.json"), 'r'))
		self.transit_info = json.load(open(f"ViCo/assets/scenes/{self.scene_name}/transit.json", 'r'))
		self.env_other_meta = ENV_OTHER_METADATA
		self.agent_names_to_group_name = {agent_name: group_name for group_name, group in self.config['groups'].items() for agent_name in group['members']} if 'groups' in self.config else {}
		self.events = EventSystem()
		self.obs = None

		self.scene_assets_dir = f"ViCo/scene/v1/{scene}"
		self.vehicles = []
		self.enable_tm_debug = enable_tm_debug

		self.scene = gs.Scene(
			# viewer_options=None,
			viewer_options=gs.options.ViewerOptions(
				res=(1000, 1000),
				camera_pos=np.array([0.0, 0.0, 1000]),
				camera_lookat=np.array([0, 0.0, 0.0]),
				camera_fov=60,
			),
			rigid_options=gs.options.RigidOptions(
				gravity=(0.0, 0.0, -9.8),
				enable_collision=self.enable_collision,
				max_collision_pairs=400
			),
			avatar_options=gs.options.AvatarOptions(
				enable_collision=self.enable_collision,
			),
			renderer=gs.renderers.Rasterizer(),
			vis_options=gs.options.VisOptions(
				show_world_frame=False,
				segmentation_level="entity",
				lights=LIGHTS
			),
			profiling_options=gs.options.ProfilingOptions(show_FPS=False),
			show_viewer=not head_less,
		)

		### Load city scene
		start_time = time.time()
		self.global_camera = self.scene.add_camera(
                res=(2048, 2048),
                pos=(0.0, 0.0, 0.0),
                lookat=(1.0, 0.0, 0.0),
                fov = 90,
                GUI=False,
            )
		terrain = self.load_city_scene(self.scene_assets_dir, no_load_scene)
		gs.logger.info(f"loading city scene took {time.time() - start_time:.2f}s")

		if no_load_scene:
			no_collision_entities = []
		else:
			no_collision_entities = [terrain]

		### Load avatars
		self.agents: list[AvatarController] = []
		self.agent_names: list[str] = self.config['agent_names']
		self.agent_infos: list[dict] = self.config['agent_infos']
		start_time = time.time()
		frame_ratio = 0.0 if skip_avatar_animation else 5.0
		for i in range(self.num_agents):
			self.agents.append(self.add_avatar(name=self.agent_names[i],
											   motion_data_path='Genesis/genesis/assets/ViCo/avatars/motions/motion.pkl',
											   skin_options={
												   'glb_path': self.config['agent_skins'][i],
												   'euler': (-90, 0, 90),
												   'pos': (0.0, 0.0, -0.959008030)
											   },
											   ego_view_options={
													"res": (self.resolution, self.resolution),
													"fov": 90,
													"GUI": False,
												},
											   frame_ratio=frame_ratio,
											   terrain_height_path=os.path.join(gs.utils.get_assets_dir(), self.scene_assets_dir, "height_field.npz"),
											   third_person_camera_resolution=128 if self.enable_third_person_cameras else None,
											   enable_collision=enable_collision))
			no_collision_entities += [self.agents[i].robot.box]

			if self.enable_third_person_cameras:
				os.makedirs(os.path.join(self.output_dir, 'tp', self.agent_names[i]), exist_ok=True)
			os.makedirs(os.path.join(self.output_dir, 'ego', self.agent_names[i]), exist_ok=True)
			os.makedirs(os.path.join(self.output_dir, 'steps', self.agent_names[i]), exist_ok=True)

		for agent in self.agents:
			agent.initialize_no_collision(no_collision_entities)
		gs.logger.info(f"loading {self.num_agents} avatars took {time.time() - start_time:.2f}s")
		### Load indoor scenes
		if self.enable_indoor_scene:
			start_time = time.time()
			for place in self.place_metadata: # loading living places
				if self.place_metadata[place]["coarse_type"] == "accommodation" and "'s room" in place:
					self.load_indoor_scene(place)
			for group in self.config['groups'].values():
				self.load_indoor_scene(group['place'])
			if 'stores' in self.config:
				for store in self.config['stores']:
					self.load_indoor_scene(store)
			gs.logger.info(f"loading {len(self.active_places_info)} indoor scenes took {time.time() - start_time:.2f}s")

			# load a default room
			self.active_places_info['default_room'], self.place_cameras['default_room'] = \
				load_default_room(self, "default_room")

		for vgeom in self.scene.rigid_solver.vgeoms:
			vgeom.surface.double_sided = True

		if no_traffic_manager:
			tm_vehicle_num = 0
			tm_avatar_num = 0
		self.traffic_manager = TrafficManager(self, scene, vehicle_number=tm_vehicle_num, pedestrian_number=tm_avatar_num, enable_tm_debug=self.enable_tm_debug, logger=gs.logger, debug=debug)

		self.scene.build()
		self.scene.reset()

		self.traffic_manager.init_post_scene_build()

		# forward pass
		if self.enable_gt_segmentation:
			# to make sure color is consistent across runs
			prev_seed = np.random.get_state()
			np.random.seed(42)
			self.entity_idx_to_color = np.random.randint(0, 255, (len(self.entity_idx_to_info) + 1, 3), dtype=np.uint8)
			# background uses black
			self.entity_idx_to_color[0, :] = 0
			np.random.set_state(prev_seed)

		self.sim_frames_per_step = int(self.sec_per_step / self.scene.dt)
		gs.logger.info(f"running {self.sim_frames_per_step} scene steps for one ViCo step of {self.sec_per_step}s")

		self.traffic_manager.reset()

		for i, agent in enumerate(self.agents):
			agent.reset(np.array(self.config['agent_poses'][i][:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(self.config['agent_poses'][i][3:], dtype=np.float64))))
			if self.config['agent_infos'][i]['current_vehicle'] == 'bus':
				agent.enter_bus(self.traffic_manager.bus.bus)
				gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bus at {self.traffic_manager.bus.bus.get_global_pose().tolist()}")
			elif self.config['agent_infos'][i]['current_vehicle'] == 'bicycle':
				bike_idx = self.enter_bike(i)
				if bike_idx is None:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot enter bike because no bike is available. This is not normal!")
				else:
					gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bicycle {bike_idx} at {self.traffic_manager.shared_bicycles.bicycles[bike_idx].get_global_pose().tolist()}")

		frame = 0
		while not all([agent.spare() for agent in self.agents]):
			frame += 1
			self.scene_step()

		gs.logger.info(f"In initialization, prepare all agents with current vehicle for {frame} frames")

		rgb_space = spaces.Box(0, 256,(3, self.resolution, self.resolution), dtype=np.int32)
		depth_space = spaces.Box(0, 256, (self.resolution, self.resolution), dtype=np.float32)
		self.observation_space_single = spaces.Dict({
			"rgb": rgb_space,
			"depth": depth_space,
			"segmentation": spaces.Box(0, 256, (self.resolution, self.resolution), dtype=np.int32),
			'fov': spaces.Box(0, 180, (1,), dtype=np.float32),
			'extrinsics': spaces.Box(-30, 30, (4, 4), dtype=np.float32),
			'pose': spaces.Box(-400, 400, (6,), dtype=np.float32), # may be larger?
			'accessible_places': spaces.Sequence(spaces.Text(max_length=1000, charset=string.printable)),
			'action_status': spaces.Text(max_length=1000, charset=string.printable),
			'current_building': spaces.Text(max_length=1000, charset=string.printable),
			'current_place': spaces.Text(max_length=1000, charset=string.printable),
			"cash": spaces.Box(0, 1000, (1,), dtype=np.int32)
		})
		self.observation_space = spaces.Dict({
			i: self.observation_space_single for i in range(self.num_agents)
		})

		self.action_space_single = spaces.Dict({
			'type': spaces.Discrete(7),
			'arg1': spaces.Text(10000),
		})
		self.action_space = spaces.Dict({
			i: self.action_space_single for i in range(self.num_agents)
		})

		self.fps_tracker = FPSTracker(0)
		# compute bounding boxes for each entity
		if len(self.entity_idx_to_info) != len(self.scene.entities):
			gs.logger.error(f"Number of entities in scene {len(self.scene.entities)} does not match number of entities in entity_idx_to_info {len(self.entity_idx_to_info)}")
		for i, e in self.entity_idx_to_info.items():
			rigid = self.scene.entities[i]
			if isinstance(rigid, RigidEntity) and e["type"] == "object":
				mx, mn = np.zeros(3), np.zeros(3)
				for geom in rigid.vgeoms:
					verts = geom._vmesh.verts
					mx = np.maximum(mx, verts.max(axis=0))
					mn = np.minimum(mn, verts.min(axis=0))
				e["bbox"] = np.stack([mn, mx])

		os.makedirs(os.path.join(self.output_dir, 'steps', 'env'), exist_ok=True)

	def add_entity(self, type, name, morph, material=None,
				   surface=None, visualize_contact=False, vis_mode=None,):
		"""
		:param type: One of "structure", "building", "object", "avatar", "avatar_box", "vehicle"
		:param name:
		:param morph:
		:param material:
		:param surface:
		:param visualize_contact:
		:param vis_mode:
		:return:
		"""
		entity = self.scene.add_entity(morph=morph, material=material, surface=surface, visualize_contact=visualize_contact, vis_mode=vis_mode)
		self.entity_idx_to_info[entity.idx] = {"type": type, "name": name}
		return entity

	def log_step_vehicle_info(self):
		step_info = {
			"curr_bus_pose": self.traffic_manager.bus.bus.get_global_pose().tolist(),
			"curr_bus_stop": self.traffic_manager.bus.current_stop_name,
			"curr_bicycle_poses": [bicycle.get_global_pose().tolist() for bicycle in self.traffic_manager.shared_bicycles.bicycles]
		}
		step_info_path = os.path.join(self.config_path.replace('curr_sim', 'steps'), 'env', f"{self.steps:06d}.json")
		atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))

	def scene_step(self):
		self.scene.step()

		if self.vehicles:
			for vehicle in self.vehicles:
				vehicle.step()
			self.log_step_vehicle_info()

		if self.agents:
			for avatar in self.agents:
				avatar.step(self.skip_avatar_animation)

		if self.traffic_manager is not None:
			for avatar in self.traffic_manager.avatars:
				avatar.avatar.step()

		if self.agents and self.enable_collision:
			collision_pairs = self.scene.rigid_solver.detect_collision()
			for avatar in self.agents:
				avatar.post_step(collision_pairs)

	def add_avatar(
			self,
			name: str,
			motion_data_path: str,
			skin_options = None,
			ego_view_options = None,
			frame_ratio = 5.0,
			terrain_height_path = None,
			third_person_camera_resolution = None,
			enable_collision = True,
	):
		avatar = AvatarController(
			env = self,
			motion_data_path = motion_data_path,
			skin_options = skin_options,
			ego_view_options = ego_view_options,
			frame_ratio = frame_ratio,
			terrain_height_path = terrain_height_path,
			third_person_camera_resolution = third_person_camera_resolution,
			enable_collision = enable_collision,
			name=name
		)
		return avatar

	def add_vehicle(
			self,
			name,
			vehicle_asset_path,
			ego_view_options,
			position = np.zeros(3, dtype=np.float64),
			rotation = np.zeros(3, dtype=np.float64),
			dt=1e-2,
			forward_speed_m_per_s=5,
			angular_speed_deg_per_s=360,
			terrain_height_path = None,
	):
		if self.skip_avatar_animation:
			dt = self.sec_per_step
		vehicle = VehicleController(
			env = self,
			name = name,
			vehicle_asset_path = vehicle_asset_path,
			ego_view_options = ego_view_options,
			position = position,
			rotation = rotation,
			dt = dt,
			forward_speed_m_per_s=forward_speed_m_per_s,
			angular_speed_deg_per_s=angular_speed_deg_per_s,
			terrain_height_path=terrain_height_path,
		)
		self.vehicles.append(vehicle)
		return vehicle

	def load_city_scene(self, scene_assets_dir, no_load_scene):
		if no_load_scene:
			return None
		terrain = self.add_entity(
			type = "structure",
			name = "terrain",
			material=gs.materials.Rigid(
				sdf_min_res=4,
				sdf_max_res=4,
			),
			morph=gs.morphs.Mesh(
				file=os.path.join(scene_assets_dir, 'terrain.glb'),
				euler=(90.0, 0, 0),
				fixed=True,
				collision=self.enable_collision,
				merge_submeshes_for_collision=False,
				group_by_material=True,
			),
		)
		buildings_dir = str(os.path.join(gs.utils.get_assets_dir(), scene_assets_dir, 'buildings'))
		building_glb2name = {}
		for building_name in self.building_metadata:
			if building_name != 'open space':
				building_glb2name[self.building_metadata[building_name]['building_glb']] = building_name
		if os.path.exists(buildings_dir):
			for building in os.listdir(buildings_dir):
				if building.endswith('.glb'):
					if building in building_glb2name:
						self.add_entity(
							type="building",
							name=building_glb2name[building],
							material=gs.materials.Rigid(
								sdf_min_res=4,
								sdf_max_res=4,
							),
							morph=gs.morphs.Mesh(
								file=os.path.join(scene_assets_dir, 'buildings', building),
								euler=(90.0, 0, 0),
								fixed=True,
								collision=self.enable_collision,
								merge_submeshes_for_collision=False,
								group_by_material=True,
								convexify=self.enable_collision,
								coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
							),
						)
					else:
						self.add_entity(
							type="structure",
							name=building,
							material=gs.materials.Rigid(
								sdf_min_res=4,
								sdf_max_res=4,
							),
							morph=gs.morphs.Mesh(
								file=os.path.join(scene_assets_dir, 'buildings', building),
								euler=(90.0, 0, 0),
								fixed=True,
								collision=self.enable_collision,
								merge_submeshes_for_collision=False,
								group_by_material=True,
								convexify=self.enable_collision,
								coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
							),
						)
		else:
			self.add_entity(
				type= "structure",
				name= "buildings",
				material=gs.materials.Rigid(
					sdf_min_res=4,
					sdf_max_res=4,
				),
				morph=gs.morphs.Mesh(
					file=os.path.join(scene_assets_dir, 'buildings.glb'),
					euler=(90.0, 0, 0),
					fixed=True,
					collision=self.enable_collision,
					merge_submeshes_for_collision=False,  # Buildings are constructed separately
					group_by_material=True,
					convexify=self.enable_collision,
					coacd_options=CoacdOptions(threshold=0.05,preprocess_resolution=200)
				),
			)

		# self.add_entity(
		# 	type="structure",
		# 	name="roof",
		# 	material=gs.materials.Rigid(
		# 		sdf_min_res=4,
		# 		sdf_max_res=4,
		# 	),
		# 	morph=gs.morphs.Mesh(
		# 		file=os.path.join(scene_assets_dir, 'roof.glb'),
		# 		euler=(90.0, 0, 0),
		# 		fixed=True,
		# 		collision=False,  # No collision needed for roof
		# 		group_by_material=True,
		# 	),
		# )

		if self.enable_outdoor_objects:
			outdoor_object_context = OutdoorObjectContext(
				scene_name=self.scene_name,
				objects_cfg_dir=os.path.join(scene_assets_dir, 'objects'),
				assets_dir='ViCo/objects/outdoor_objects',
				max_objects=self.outdoor_objects_max_num,
				seed=self.seed,
				terrain_height_field_path=f"{scene_assets_dir}/height_field.npz",
				road_info_path=f"ViCo/assets/scenes/{self.scene_name}/road_data/roads.pkl",
			)
			load_outdoor_objects(self, outdoor_object_context, self.transit_info)

		return terrain

	def enter_bike(self, agent_idx):
		nearest_bicycle, nearest_bicycle_idx = self.traffic_manager.shared_bicycles.get_nearest_bicycle(self.agents[agent_idx].get_global_xy())
		if nearest_bicycle is not None:
			self.agents[agent_idx].enter_bike(0, nearest_bicycle)
			self.traffic_manager.shared_bicycles.start_timer(nearest_bicycle_idx, self.curr_time)
			self.agent_infos[agent_idx]["current_vehicle"] = "bicycle"
		return nearest_bicycle_idx

	def exit_bike(self, agent_idx):
		if self.agent_infos[agent_idx]["current_vehicle"] != "bicycle":
			gs.logger.warning(f"Agent {self.agent_names[agent_idx]} cannot exit bike because current vehicle is not bike.")
			return False
		nearest_bicycle, nearest_bicycle_idx = self.traffic_manager.shared_bicycles.get_riding_bicycle(self.agents[agent_idx].get_global_xy())
		cost = self.traffic_manager.shared_bicycles.end_timer(nearest_bicycle_idx, self.curr_time)
		self.agent_infos[agent_idx]["cash"] -= cost
		self.agent_infos[agent_idx]["current_vehicle"] = None
		self.agents[agent_idx].exit_bike(0)
		return True

	def step(self, agent_actions):
		for i, agent in enumerate(self.agents):
			gs.logger.debug(f"Agent {self.agent_names[i]} with state {agent.robot.base_state}.")
			action = agent_actions[i]
			if action is None:
				continue
			agent.robot.action_status = ActionStatus.SUCCEED
			if action['type'] == 'move_forward':
				agent.move_forward(action['arg1'], self.sec_per_step * 1.0)
			elif action['type'] == 'teleport':
				agent.reset_with_global_xy(np.array(action['arg1']))
			elif action['type'] == 'turn_left':
				agent.turn_left(action['arg1'], turn_sec_limit=self.sec_per_step * 1500)
			elif action['type'] == 'turn_right':
				agent.turn_right(action['arg1'], turn_sec_limit=self.sec_per_step * 1500)
			elif action['type'] == 'look_at':
				target_pos = action['arg1']
				## make avatar look at target_pos by turn_left or turn_right
				agent_pos = agent.robot.global_trans
				agent_rot = agent.robot.global_rot
				agent_dir = agent_rot[:, 0]
				target_dir = target_pos - agent_pos	
				agent_dir[2] = 0
				target_dir[2] = 0
				agent_dir = agent_dir / np.linalg.norm(agent_dir)
				target_dir = target_dir / np.linalg.norm(target_dir)
				cross = np.cross(agent_dir, target_dir)
				dot = np.dot(agent_dir, target_dir)
				angle = np.arccos(dot)
				if cross[2] > 0:
					agent.turn_left(angle, turn_sec_limit=self.sec_per_step * 1500)
				else:
					agent.turn_right(angle, turn_sec_limit=self.sec_per_step * 1500)

			elif action['type'] == 'sleep':
				agent.sleep()
			elif action['type'] == 'wake':
				agent.wake()
			elif action['type'] == 'enter' or action['type'] == 'force_enter':
				if action['type'] == 'force_enter' or action['arg1'] in self.obs[i]['accessible_places']:
					if action['arg1'] == 'open space':
						if self.agent_infos[i]["current_place"] in self.active_places_agents:
							try:
								self.active_places_agents[self.agent_infos[i]["current_place"]].remove(self.agent_names[i])
							except ValueError:
								gs.logger.warning(f"Agent {self.agent_names[i]} is not among the names that are in place {self.agent_infos[i]['current_place']}.")
						agent.reset(np.array(self.agent_infos[i]["outdoor_pose"][:3]), geom_utils.euler_to_R(np.degrees(np.array(self.agent_infos[i]["outdoor_pose"][3:], dtype=np.float64))))
						self.agent_infos[i]["current_building"] = 'open space'
						self.agent_infos[i]["current_place"] = None
					else:
						if self.place_metadata[action['arg1']]['building'] == 'open space':
							continue
						if self.agent_infos[i]["current_building"] == 'open space':
							self.agent_infos[i]["outdoor_pose"] = self.config['agent_poses'][i]
						self.load_indoor_scene(action['arg1']) # load new scenes should be wrong now
						if "init_avatar_poses" in self.active_places_info[action['arg1']]:
							pos = self.active_places_info[action['arg1']]["init_avatar_poses"][0]["pos"]
							euler = self.active_places_info[action['arg1']]["init_avatar_poses"][0]["euler"]
							x, y, z = self.place_metadata[action['arg1']]['location']
							pos = np.array([pos[0] + x, pos[1] + y, z])
							agent.reset(pos, geom_utils.euler_to_R(np.degrees(np.array(euler, dtype=np.float64))))
						else:
							offset = len(self.active_places_agents[action['arg1']]) * 0.2
							self.active_places_agents[action['arg1']].append(self.agent_names[i])
							x, y, z = self.place_metadata[action['arg1']]['location']
							agent.reset(np.array([x + offset, y + offset, z]), geom_utils.euler_to_R(np.degrees(np.array([0, 0, 0])))) # elevator entrance
						self.agent_infos[i]["current_building"] = self.place_metadata[action['arg1']]['building']
						self.agent_infos[i]["current_place"] = action['arg1']

				else:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot enter {action['arg1']} because it is not in accessible places.")
					agent.robot.action_status = ActionStatus.FAIL
			elif action['type'] == 'enter_bus':
				if "bus" not in self.obs[i]['accessible_places']:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot enter bus because bus at {self.traffic_manager.bus.bus.get_global_pose().tolist()} is not in accessible places.")
					agent.robot.action_status = ActionStatus.FAIL
				agent.enter_bus(self.traffic_manager.bus.bus)
				self.agent_infos[i]["current_vehicle"] = "bus"
			elif action['type'] == 'exit_bus':
				if self.agent_infos[i]["current_vehicle"] != "bus":
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot exit bus because current vehicle is not bus.")
					agent.robot.action_status = ActionStatus.FAIL
				agent.exit_bus()
				self.agent_infos[i]["current_vehicle"] = None
			elif action['type'] == 'enter_bike':
				if not self.enter_bike(i):
					agent.robot.action_status = ActionStatus.FAIL
			elif action['type'] == 'exit_bike':
				if not self.exit_bike(i):
					agent.robot.action_status = ActionStatus.FAIL
			elif action['type'] == 'pick': # arg1: hand id [0,1], arg2: position
				pos = np.array(action['arg2'])
				min_volume, entity_idx = 1e10, None
				for j, e in self.entity_idx_to_info.items():
					if "bbox" in e:
						bbox = e["bbox"]
						rigid: RigidEntity = self.entities[j]
						rel_pos = pos - rigid.get_pos().cpu().numpy()
						if np.all(rel_pos > bbox[0] - 0.02) and np.all(rel_pos < bbox[1] + 0.02):
							volume = np.prod(bbox[1] - bbox[0])
							if volume < min_volume:
								min_volume, entity_idx = volume, j

				if entity_idx is None:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot pick at {pos} because no entity is found.")
					agent.robot.action_status = ActionStatus.FAIL
					continue
				self.agent_infos[i]["held_objects"][action['arg1']] = self.entities[entity_idx]["name"]
				agent.pick(action['arg1'], self.entities[entity_idx])
			elif action['type'] == 'put': # arg1: hand id [0,1]
				hand_id = action['arg1']
				entity = agent.robot.attached_object[hand_id]
				agent.put(action['arg1'], action.get('arg2', None))
			elif action['type'] == 'stand':
				agent.stand_up()
			elif action['type'] == 'sit':
				agent.sit(position=np.array(action['arg1'][0]))
			elif action['type'] == 'drink':
				agent.drink(action['arg1'])
			elif action['type'] == 'eat':
				agent.eat(action['arg1'])
			elif action['type'] == 'exchange': # arg1: target agent name, arg2: amount
				if agent.robot.base_state == AvatarState.SLEEPING:
					agent.robot.base_state = AvatarState.STANDING
				if action['arg2'] > self.agent_infos[i]["cash"]:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot exchange {action['arg2']} cash with {action['arg1']} because it does not have enough cash.")
					agent.robot.action_status = ActionStatus.FAIL
					continue
				target_agent_idx = self.agent_names.index(action['arg1'])
				self.agent_infos[i]["cash"] -= action['arg2']
				self.agent_infos[target_agent_idx]["cash"] += action['arg2']
			elif action['type'] == 'converse':
				if agent.robot.base_state == AvatarState.SLEEPING:
					agent.robot.base_state = AvatarState.STANDING
				agent_pos = self.config['agent_poses'][i][:3]
				converse_range = action['arg2'] if 'arg2' in action else 10
				priority = random.randint(0, 100)
				if converse_range > 10:
					gs.logger.warning(f"Agent {self.agent_names[i]} attempted to converse with range {converse_range} which is larger than 10. Ignored.")
					self.agents[i].robot.action_status = ActionStatus.FAIL
					continue
				deleted_subjects = self.events.add(type="speech", pos=agent_pos, r=converse_range, content=action['arg1'], priority=priority, subject=self.agent_names[i], predicate="is", object="talk")
				# if interleaved with other speech events, keep only the highest priority one, drop others and give it fail
				for deleted_subject in deleted_subjects:
					self.agents[self.agent_names.index(deleted_subject)].robot.action_status = ActionStatus.FAIL
			elif action['type'] == 'wait':
				continue
			elif action['type'] == 'task_complete':
				continue
			else:
				raise NotImplementedError(f"agent action type {action['type']} is not supported")

		if self.defer_chat:
			# post-generate utterances for remained speech events
			start_time = time.perf_counter()
			for idx, event in self.events.events.items():
				if event["type"] == "speech":
					agent_id = self.agent_names.index(event["subject"])
					agent_actions[agent_id]['request_chat_func'](event["content"])
			to_delete_id = []
			for idx, event in self.events.events.items():
				if event["type"] == "speech":
					agent_id = self.agent_names.index(event["subject"])
					event["content"] = agent_actions[agent_id]['get_utterance_func'](self.steps)
					if event["content"] is None:
						to_delete_id.append(idx)
			self.events.delete(to_delete_id)

			dt_chat = time.perf_counter() - start_time
			self.config["dt_chat"] = (self.config["dt_chat"] * self.steps + dt_chat) / (self.steps + 1)

		if self.traffic_manager is not None:
			self.traffic_manager.step()

		if self.skip_avatar_animation:
			sim_early_end = True
			self.scene_step()
		else:
			sim_early_end = False
			frames = 0
			for _ in tqdm.tqdm(range(self.sim_frames_per_step), desc="simulating", ):
				# need to advance at least one frame to update agent position in the case of reset
				self.scene_step()
				frames += 1
				if all([agent.spare() for agent in self.agents]) and self.traffic_manager.spare():
					sim_early_end = True
					gs.logger.info(f"After {frames} frames, all agents finished action, end simulation early.")
					break
		self.traffic_manager.bus.step(self.curr_time)
		self.steps += 1
		self.seconds += self.sec_per_step
		self.curr_time += timedelta(seconds=self.sec_per_step)

		if sim_early_end and not self.traffic_manager.bus.stop_at_this_step:
			bus_next_pose = self.traffic_manager.bus.update_at_time(self.curr_time)
			self.traffic_manager.bus.reset(np.array(bus_next_pose[:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(bus_next_pose[3:], dtype=np.float64))))
			# to update the position of agents in bus
			self.scene_step()

		self.config['step'] = self.steps
		self.config['curr_time'] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		self.config['agent_poses'] = []

		for i, agent in enumerate(self.agents):
			self.config['agent_poses'].append(agent.get_global_pose().tolist())
		self.config['agent_infos'] = self.agent_infos
		for i in range(0, len(self.config["bicycle_poses"])):
			self.config["bicycle_poses"][i] = self.traffic_manager.shared_bicycles.bicycles[i].get_global_pose().tolist()
		atomic_save(os.path.join(self.config_path, 'config.json'), json.dumps(self.config, indent=4, default=json_converter))
		start_time = time.perf_counter()
		self.obs = self.get_obs()
		dt_obs = time.perf_counter() - start_time
		self.fps_tracker.step()
		return self.obs, 0, False, {}

	def reset(self):
		self.scene.reset()
		self.traffic_manager.reset()
		for i, agent in enumerate(self.agents):
			agent.reset(np.array(self.config['agent_poses'][i][:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(self.config['agent_poses'][i][3:], dtype=np.float64))))
			if self.config['agent_infos'][i]['current_vehicle'] == 'bus':
				agent.enter_bus(self.traffic_manager.bus.bus)
				gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bus at {self.traffic_manager.bus.bus.get_global_pose().tolist()}")
			elif self.config['agent_infos'][i]['current_vehicle'] == 'bicycle':
				bike_idx = self.enter_bike(i) # how to resume previous cost?
				if bike_idx is None:
					gs.logger.warning(f"Agent {self.agent_names[i]} cannot enter bike because no bike is available. This is not normal!")
				else:
					gs.logger.info(f"In initialization, Agent {self.agent_names[i]} at {agent.get_global_pose().tolist()} enters bicycle {bike_idx} at {self.traffic_manager.shared_bicycles.bicycles[bike_idx].get_global_pose().tolist()}")

		self.scene_step()
		self.steps = self.config['step']
		self.seconds = self.steps * self.sec_per_step
		self.curr_time = datetime.strptime(self.config['curr_time'], "%B %d, %Y, %H:%M:%S")
		self.obs = self.get_obs()
		self.fps_tracker = FPSTracker(0)
		return self.obs

	def set_curr_time(self, curr_time):
		gs.logger.warning(f"Set current time to {curr_time}")
		self.curr_time = curr_time
		self.config['curr_time'] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		atomic_save(os.path.join(self.config_path, 'config.json'), json.dumps(self.config, indent=4, default=json_converter))
		# bus_next_pose = self.traffic_manager.bus.update_at_time(self.curr_time)
		# self.traffic_manager.bus.reset(np.array(bus_next_pose[:3], dtype=np.float64), geom_utils.euler_to_R(np.degrees(np.array(bus_next_pose[3:], dtype=np.float64))))
		self.scene_step()
		self.obs = self.get_obs()
		return self.obs

	def get_obs(self):
		if self.enable_third_person_cameras and self.seconds % self.save_per_seconds == 0:
			for i, agent in enumerate(self.agents):
				indoor = self.agent_infos[i]['current_building'] != 'open space'
				third_person_rgb = agent.get_third_person_camera_rgb(indoor)
				if third_person_rgb is not None:
					Image.fromarray(third_person_rgb).save(os.path.join(self.output_dir, 'tp', self.agent_names[i], f"rgb_{self.steps:06d}.png"))

		if self.traffic_manager is not None and self.enable_tm_debug and self.seconds % self.save_per_seconds == 0:
			for i, avatar in enumerate(self.traffic_manager.avatars):
				if avatar.avatar.ego_view is not None:
					third_person_rgb = avatar.avatar.get_third_person_camera_rgb()
					Image.fromarray(third_person_rgb).save(os.path.join(self.output_dir, 'traffic_ego', avatar.name, f"rgb_avatar{str(i)}_{avatar.name}_{self.steps:06d}.png"))
			for i, vehicle in enumerate(self.traffic_manager.vehicles):
				if vehicle.vehicle.ego_view is not None:
					rgb, _, _, _, _ = vehicle.vehicle.render_ego_view(depth=True, segmentation=self.enable_gt_segmentation)
					Image.fromarray(rgb).save(os.path.join(self.output_dir, 'traffic_ego', vehicle.vehicle.name, f"rgb_vehicle{str(i)}_{vehicle.vehicle.name}_{self.steps:06d}.png"))

		obs = {i: {} for i in range(self.num_agents)}
		for i, agent in enumerate(self.agents):
			obs[i]['rgb'], obs[i]['depth'], obs[i]['segmentation'], obs[i]['fov'], obs[i]['extrinsics'] = agent.render_ego_view(depth=True, segmentation=self.enable_gt_segmentation)
			obs[i]['pose'] = self.config['agent_poses'][i]
			if self.seconds % self.save_per_seconds == 0:
				if obs[i]['rgb'] is not None:
					Image.fromarray(obs[i]['rgb']).save(os.path.join(self.output_dir, 'ego', self.agent_names[i], f"rgb_{self.steps:06d}.png"))
				if self.debug:
					# if obs[i]['depth'] is not None:
					# 	Image.fromarray(100 / obs[i]['depth']).convert('RGB').save(os.path.join(self.output_dir, 'ego', self.agent_names[i], f"depth_{self.steps:06d}.png"))
					if obs[i]['segmentation'] is not None:
						Image.fromarray(self.entity_idx_to_color[obs[i]['segmentation'] + 1]).save(os.path.join(self.output_dir, 'ego', self.agent_names[i], f"seg_{self.steps:06d}.png"))
			obs[i]['events'] = self.events.get(ref_pos=(obs[i]['pose'][:3])) if self.challenge != 'commute' else []

			if not self.enable_gt_segmentation:
				for event in obs[i]['events']:
					event['subject'] = None
					event['predicate'] = None
					event['object'] = None
			obs[i]['curr_time'] = self.curr_time
			obs[i]['steps'] = self.steps
			obs[i]['accessible_places'] = get_accessible_places(self.building_metadata, self.place_metadata, agent.get_global_pose(), self.agent_infos[i]["current_building"], self.agent_infos[i]["current_place"])
			obs[i]['action_status'] = agent.action_status().value
			obs[i]['current_building'] = self.agent_infos[i]["current_building"]
			obs[i]['current_place'] = self.agent_infos[i]["current_place"]
			obs[i]['cash'] = self.agent_infos[i]["cash"]
			obs[i]['held_objects'] = self.agent_infos[i]["held_objects"] if "held_objects" in self.agent_infos[i] else [None, None]
			obs[i]['current_vehicle'] = self.agent_infos[i]["current_vehicle"]

			if obs[i]['current_vehicle'] == 'bus':
				if self.traffic_manager.bus.stop_at_this_step:
					obs[i]['accessible_places'] = [self.traffic_manager.bus.current_stop_name]
				else:
					obs[i]['accessible_places'] = []
			elif obs[i]['current_vehicle'] == 'bicycle':
				#remove the indoor places from accessible places
				obs[i]['accessible_places'] = [place for place in obs[i]['accessible_places'] if self.place_metadata[place]['building'] == 'open space']
			else:
				if self.traffic_manager.bus.stop_at_this_step and is_near_goal(obs[i]['pose'][0], obs[i]['pose'][1], None, self.traffic_manager.bus.bus.get_global_pose().tolist()[:2], threshold=8):
					obs[i]['accessible_places'].append('bus')
				nearest_bicycle, nearest_bicycle_idx = self.traffic_manager.shared_bicycles.get_nearest_bicycle(agent.get_global_xy())
				if nearest_bicycle is not None:
					obs[i]['accessible_places'].append('bicycle')
			if self.enable_gt_segmentation:
				obs[i]["gt_seg_entity_idx_to_info"] = self.entity_idx_to_info
		self.events.clear()
		return obs

	def close(self):
		gs.logger.warning("Close ViCo environment")
		import gc
		self.scene = None
		gc.collect()
		gs.destroy()

	def load_indoor_scene(self, place_name):
		if not self.enable_indoor_scene or not place_name:
			return

		place = self.place_metadata.get(place_name)
		if not place or 'scene' not in place:
			gs.logger.error(f"Place {place_name} is not in the metadata or missing 'scene'. Skipping loading.")
			return

		if place.get('building') == 'open space' or not place.get('scene'):
			return

		if place_name in self.active_places_info:
			gs.logger.info(f"Place {place_name} is already loaded. Skipping loading.")
			return

		gs.logger.info(f"Loading indoor scene for place {place_name}")
		
		if self.scene.is_built:
			gs.logger.warning("Scene is already built. Redirect to the default scene.")
			self.active_places_info[place_name] = self.active_places_info["default_room"]
			self.place_cameras[place_name] = self.place_cameras["default_room"]
			place['location'] = [1500, 1500, -50]
			return
		self.active_places_info[place_name], self.place_cameras[place_name] = \
				load_indoor_room(self, place, place_name, self.enable_indoor_objects)

	@property
	def entities(self):
		"""All the entities in the scene."""
		return self.scene.entities

class AgentProcess(mp.Process):
	def __init__(self, agent_cls: type[Agent], name: str, **kwargs: dict):
		super().__init__(daemon=True)
		self.agent_cls = agent_cls
		self.name = name
		self.logging_level: str = kwargs.pop("logging_level", "INFO")
		self.multi_process: bool = kwargs.pop("multi_process", True)
		self.sim_path: str = kwargs.get("sim_path", "curr_sim")
		self.kwargs = kwargs
		self.input_queue = mp.Queue()
		self.action_queue = mp.Queue()
		self.utterance_queue = mp.Queue()
		if not self.multi_process:
			self.agent = self.create_agent()
		else:
			self.agent = None

	def create_agent(self):
		logger = AgentLogger(self.name, self.logging_level,
							 os.path.join(self.sim_path.replace('curr_sim', 'logs'), f"{self.name}.log"))
		agent = self.agent_cls(self.name, logger=logger, **self.kwargs)
		return agent

	def log_step_agent_info(self, obs_i, action):
		obs_printable = {k: v for k, v in obs_i.items() if
						 not isinstance(v, np.ndarray) and not isinstance(v, datetime)}
		obs_printable.pop("gt_seg_entity_idx_to_info", None)
		step_info = {"curr_time": obs_i["curr_time"],
					 "obs": obs_printable,
					 "action": action,  # todo: if ongoing, then log down the last action [which is ongoing]
					 "curr_events": self.agent.curr_events if hasattr(self.agent, "curr_events") else None,
					 "react_mode": self.agent.react_mode if hasattr(self.agent, "react_mode") else None,
					 "chatting_buffer": self.agent.chatting_buffer if hasattr(self.agent, "chatting_buffer") else None,
					 "commute_plan": self.agent.commute_plan if hasattr(self.agent, "commute_plan") else None,
					 "commute_plan_idx": self.agent.commute_plan_idx if hasattr(self.agent, "commute_plan_idx") else None, }
		if hasattr(self.agent, "curr_goal_description"):
			step_info["action_desp"] = self.agent.curr_goal_description
			step_info["action_dura"] = self.agent.curr_goal_duration
			step_info["action_location"] = self.agent.curr_goal_address
		elif hasattr(self.agent, "hourly_schedule") and self.agent.curr_schedule_idx < len(self.agent.hourly_schedule):
			step_info["action_desp"] = self.agent.hourly_schedule[self.agent.curr_schedule_idx]["activity"]
			step_info["action_location"] = self.agent.hourly_schedule[self.agent.curr_schedule_idx]["place"] if "place" in \
																												self.agent.hourly_schedule[
																													self.agent.curr_schedule_idx] else None

		step_info_path = os.path.join(self.sim_path.replace('curr_sim', 'steps'), self.name, f"{self.agent.steps:06d}.json")
		atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))

	def run(self):
		self.agent = self.create_agent()
		self.agent.logger.info(f"Agent {self.name} started")
		while True:
			input = self.input_queue.get()
			if input["type"] == "chat":
				content = input["content"]
				try:
					utterance = self.agent.chat(content)
				except Exception as e:
					self.agent.logger.critical(
						f"Agent {self.name} generated an exception while generating utterance: {e} with traceback: {traceback.format_exc()}")
					if "Stop the exp immediately" in e.args[0]:
						break # end the agent process immediately
					utterance = None
				self.utterance_queue.put(utterance)
				step_info_path = os.path.join(self.sim_path.replace('curr_sim', 'steps'), self.name,
											  f"{self.agent.steps:06d}.json")
				step_info = json.load(open(step_info_path, 'r'))
				step_info["action"]["arg1"] = utterance
				step_info["chatting_buffer"] = self.agent.chatting_buffer if hasattr(self.agent, "chatting_buffer") else None
				atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))
			elif input["type"] == "obs":
				obs = input["content"]
				self.agent.logger.debug(f"Agent {self.name} received obs at {obs['curr_time']}")
				try:
					action = self.agent.act(obs)
				except Exception as e:
					self.agent.logger.critical(
						f"Agent {self.name} generated an exception: {e} with traceback: {traceback.format_exc()}")
					action = None
				self.action_queue.put(action)
				self.log_step_agent_info(obs, action)
			else:
				self.agent.logger.error(f"Agent {self.name} received unknown input: {input}")

	def update(self, obs):
		self.input_queue.put({"type": "obs", "content": obs})

	def act(self):
		if self.multi_process:
			return self.action_queue.get()
		else:
			obs = self.input_queue.get()["content"]
			try:
				action = self.agent.act(obs)
			except Exception as e:
				self.agent.logger.error(
					f"Agent {self.name} generated an exception: {e} with traceback: {traceback.format_exc()}")
				action = None
			self.log_step_agent_info(obs, action)
			return action

	def request_chat(self, content):
		self.input_queue.put({"type": "chat", "content": content})

	def get_utterance(self, steps):
		if self.multi_process:
			utterance = self.utterance_queue.get()
		else:
			content = self.input_queue.get()["content"]
			try:
				utterance = self.agent.chat(content)
			except Exception as e:
				self.agent.logger.error(
					f"Agent {self.name} generated an exception while generating utterance: {e} with traceback: {traceback.format_exc()}")
				utterance = None

		return utterance


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--precision", type=str, default='32')
	parser.add_argument("--logging_level", type=str, default='info')
	parser.add_argument("--backend", type=str, default='gpu')
	parser.add_argument("--head_less", '-l', action='store_true')
	parser.add_argument("--multi_process", '-m', action='store_true')
	parser.add_argument("--model_server_port", type=int, default=0)
	parser.add_argument("--output_dir", "-o", type=str, default='output')
	parser.add_argument("--debug", action='store_true')
	parser.add_argument("--overwrite", action='store_true')
	parser.add_argument("--challenge", type=str, default='full')

	### Simulation configurations
	parser.add_argument("--resolution", type=int, default=512)
	parser.add_argument("--enable_collision", action='store_true')
	parser.add_argument("--skip_avatar_animation", action='store_true')
	parser.add_argument("--enable_gt_segmentation", action='store_true')
	parser.add_argument("--max_seconds", type=int, default=86400) # 24 hours
	parser.add_argument("--save_per_seconds", type=int, default=10)
	parser.add_argument("--enable_third_person_cameras", action='store_true')
	parser.add_argument("--curr_time", type=str)

	### Scene configurations
	parser.add_argument("--scene", type=str, default='NY')
	parser.add_argument("--no_load_indoor_scene", action='store_true')
	parser.add_argument("--no_load_indoor_objects", action='store_true')
	parser.add_argument("--no_load_outdoor_objects", action='store_true')
	parser.add_argument("--outdoor_objects_max_num", type=int, default=10)
	parser.add_argument("--no_load_scene", action='store_true')

	# Traffic configurations
	parser.add_argument("--no_traffic_manager", action='store_true')
	parser.add_argument("--tm_vehicle_num", type=int, default=0)
	parser.add_argument("--tm_avatar_num", type=int, default=0)
	parser.add_argument("--enable_tm_debug", action='store_true')

	### Agent configurations
	parser.add_argument("--num_agents", type=int, default=15)
	parser.add_argument("--config", type=str, default='agents_num_15')
	parser.add_argument("--agent_type", type=str, choices=['tour_agent'], default='tour_agent')
	parser.add_argument("--detect_interval", type=int, default=1)

	args = parser.parse_args()

	args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")

	if args.overwrite and os.path.exists(args.output_dir):
		print(f"Overwrite the output directory: {args.output_dir}")
		shutil.rmtree(args.output_dir)
	os.makedirs(args.output_dir, exist_ok=True)
	config_path = os.path.join(args.output_dir, 'curr_sim')
	if not os.path.exists(config_path):
		seed_config_path = os.path.join('ViCo/assets/scenes', args.scene, args.config)
		print(f"Initiate new simulation from config: {seed_config_path}")
		try:
			shutil.copytree(seed_config_path, config_path)
		except OSError as exc:
			if exc.errno in (errno.ENOTDIR, errno.EINVAL):
				shutil.copy(seed_config_path, config_path)
			else:
				raise
	else:
		print(f"Continue simulation from config: {config_path}")

	if args.debug:
		args.enable_third_person_cameras = True
		if args.curr_time is not None:
			config = json.load(open(os.path.join(config_path, 'config.json'), 'r'))
			config['curr_time'] = args.curr_time
			atomic_save(os.path.join(config_path, 'config.json'), json.dumps(config, indent=4, default=json_converter))

	os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)

	env = VicoEnv(
		seed=args.seed,
		precision=args.precision,
		logging_level=args.logging_level,
		backend= gs.cpu if args.backend == 'cpu' else gs.gpu,
		head_less=args.head_less,
		resolution=args.resolution,
		challenge=args.challenge,
		num_agents=args.num_agents,
		config_path=config_path,
		scene=args.scene,
		enable_indoor_scene=not args.no_load_indoor_scene,
		enable_outdoor_objects=not args.no_load_outdoor_objects,
		outdoor_objects_max_num=args.outdoor_objects_max_num,
		enable_collision=args.enable_collision,
		skip_avatar_animation=args.skip_avatar_animation,
		enable_gt_segmentation=args.enable_gt_segmentation,
		no_load_scene=args.no_load_scene,
		output_dir=args.output_dir,
		enable_third_person_cameras=args.enable_third_person_cameras,
		no_traffic_manager=args.no_traffic_manager,
		enable_tm_debug=args.enable_tm_debug,
		tm_vehicle_num=args.tm_vehicle_num,
		tm_avatar_num=args.tm_avatar_num,
		save_per_seconds=args.save_per_seconds,
		defer_chat=True,
		debug=args.debug,
		enable_indoor_objects=not args.no_load_indoor_objects,
	)

	agents = []
	for i in range(args.num_agents):
		basic_kwargs = dict(
			name = env.agent_names[i],
			pose = env.config["agent_poses"][i],
			info = env.agent_infos[i],
			sim_path = config_path,
			debug = args.debug,
			logging_level = args.logging_level,
			multi_process = args.multi_process
		)
		if args.agent_type == 'tour_agent':
			from agents import TourAgent
			agents.append(AgentProcess(TourAgent, **basic_kwargs, tour_spatial_memory=env.building_metadata))
		else:
			raise NotImplementedError(f"agent type {args.agent_type} is not supported")

	if args.multi_process:
		gs.logger.info("Start agent processes")
		for agent in agents:
			agent.start()
		gs.logger.info("Agent processes started")

	# Simulation loop
	obs = env.reset()
	agent_actions = {}
	agent_actions_to_print = {}
	args.max_steps = args.max_seconds // env.sec_per_step
	while True:
		lst_time = time.perf_counter()
		for i, agent in enumerate(agents):
			agent.update(obs[i])
		for i, agent in enumerate(agents):
			agent_actions[i] = agent.act()
			agent_actions_to_print[agent.name] = agent_actions[i]['type'] if agent_actions[i] is not None else None
			if agent_actions[i] is not None and agent_actions[i]['type'] == 'converse':
				agent_actions[i]['request_chat_func'] = agent.request_chat
				agent_actions[i]['get_utterance_func'] = agent.get_utterance

		gs.logger.info(f"current time: {env.curr_time}, ViCo steps: {env.steps}, agents actions: {agent_actions_to_print}")
		dt_agent = time.perf_counter() - lst_time
		env.config["dt_agent"] = (env.config["dt_agent"] * env.steps + dt_agent) / (env.steps + 1)
		lst_time = time.perf_counter()
		obs, _, done, info = env.step(agent_actions)

		dt_sim = time.perf_counter() - lst_time
		env.config["dt_sim"] = (env.config["dt_sim"] * (env.steps - 1) + dt_sim) / env.steps
		gs.logger.info(f"Time used: {dt_agent:.2f}s for agents, {dt_sim:.2f}s for simulation, "
					   f"average {env.config['dt_agent']:.2f}s for agents, "
					   f"{env.config['dt_sim']:.2f}s for simulation, "
					   f"{env.config['dt_chat']:.2f}s for post-chatting over {env.steps} steps.")
		if env.steps > args.max_steps:
			break

	for agent in agents:
		agent.terminate()
	env.close()
