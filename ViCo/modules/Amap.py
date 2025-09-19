import ast
import os
import pdb
import random
import copy
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import pickle
import re
from enum import Enum
import time
import math
import heapq
import matplotlib.pyplot as plt
import argparse

if __name__ != "__main__" :
    from ViCo.tools.utils import *
    from ViCo.modules import *

def is_point_enclosed_Amap(grid, point, resolution, min_x, min_y, nx, ny):
    from collections import deque

    i = int((point[0] - min_x) / resolution)
    j = int((point[1] - min_y) / resolution)

    if i < 0 or i >= nx or j < 0 or j >= ny:
        # print("Point is out of bounds")
        return True, (i, j) # not valid

    if grid[i, j] == 1:
        # print("Point is inside an obstacle")
        return True, (i, j) # not valid
    else:
        return False, (i, j)

    visited = np.zeros_like(grid, dtype=bool)
    queue = deque()
    queue.append((i, j))
    visited[i, j] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        if x == 0 or x == nx - 1 or y == 0 or y == ny - 1:
            # print("Point is not enclosed")
            return False, (i, j)
        for dx, dy in directions:
            nx_ = x + dx
            ny_ = y + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny:
                if not visited[nx_, ny_] and grid[nx_, ny_] == 0:
                    visited[nx_, ny_] = True
                    queue.append((nx_, ny_))
    # print("Point is enclosed")
    return True, (i, j)

@dataclass
class Waypoints:
    id: int
    name: str | None = None
    location: list[float, float] | None = None
    belong: str | None = None
    predecessor: list = field(default_factory=list)
    successor: list = field(default_factory=list)

class Route:
    def __init__(self, waypoints=None):
        '''
        waypoints: list(np.array([int,int]))
        '''
        self.waypoints=waypoints

    def __getitem__(self, key):
        # Slice the waypoints using the provided key (can be int or slice)
        sliced_waypoints = self.waypoints[key]
        # Return a new Route object with the sliced waypoints
        return Route(sliced_waypoints)
    
    def __len__(self):
        return len(self.waypoints)
    
    def empty(self):
        return not self.waypoints

    def calc_time(self, pose=None):
        if pose is not None:
            ret=np.linalg.norm(np.array(self.waypoints[0][:2])-np.array(pose[:2]))
        for i in range(1, len(self.waypoints)):
            ret+=np.linalg.norm(np.array(self.waypoints[i][:2])-np.array(self.waypoints[i-1][:2]))
        return ret*2 # for turning

class Amap:
    '''walkers only'''
    def __init__(self, scene_name=None, pose=None, place_metadata=None, building_metadata=None, waypoints_dis=7., logger=None):
        self.scene_name=scene_name
        self.pose=pose
        self.covered_length=0.
        self.place_metadata=deepcopy(place_metadata)
        self.building_metadata=deepcopy(building_metadata)
        self.waypoints_dis=waypoints_dis

        with open(f'ViCo/assets/scenes/{scene_name}/raw/center.txt', "r") as file:
            for line in file:
                ref_lat, ref_lon = line.strip().split()
            ref_lat, ref_lon = float(ref_lat), float(ref_lon)
        # self.map = LocalMap(file_path=f"ViCo/assets/scenes/{scene_name}/road_data/road_data.xodr", terrain_height_path=None, ref_lat=ref_lat, ref_lon=ref_lon)
        self.roads, self.nodes = pickle.load(open(f"ViCo/assets/scenes/{scene_name}/road_data/roads.pkl", 'rb'))

        obstacle_grid_save = pickle.load(open(f"ViCo/assets/scenes/{scene_name}/obstacle_grid.pkl", 'rb'))
        self.obstacle_grid = obstacle_grid_save["grid"]
        self.obstacle_grid_parameters = obstacle_grid_save["parameters"]

        self.waypoints = []
        self.road2waypoint = {}
        self.spawn_waypoints()

        self.logger = logger
        
    def reset(self, pose):
        self.pose=pose
        self.covered_length=0.

    def is_point_invalid(self, point):
        return all([is_point_enclosed_Amap(grid=self.obstacle_grid, point=point+shift, resolution=self.obstacle_grid_parameters["resolution"], min_x=self.obstacle_grid_parameters["min_x"], min_y=self.obstacle_grid_parameters["min_y"], nx=self.obstacle_grid_parameters["nx"], ny=self.obstacle_grid_parameters["ny"])[0] for shift in [np.array([i, j]) for i in range(-int(self.waypoints_dis),int(self.waypoints_dis)+1) for j in range(-int(self.waypoints_dis),int(self.waypoints_dis)+1)]])

    def spawn_waypoints(self):
        for node in self.nodes:
            # if the point is invalid
            if self.is_point_invalid([self.nodes[node]['x'], self.nodes[node]['y']]): continue
            # processing node
            for road in self.nodes[node]["connected_roads"]:
                self.road2waypoint[road]=len(self.waypoints)
            self.nodes[node]["2wp"]=len(self.waypoints)
            self.waypoints.append(Waypoints(id=len(self.waypoints), location=[self.nodes[node]['x'], self.nodes[node]['y']], belong=None if not self.nodes[node]["connected_roads"] else self.nodes[node]["connected_roads"][0]))
        for road in self.roads:
            # spawn points on road
            start_x, start_y = road['start']['x'],road['start']['y']
            end_x, end_y = road['end']['x'],road['end']['y']
            length = np.linalg.norm(np.array([start_x-end_x, start_y-end_y]))
            s=self.waypoints_dis
            # if the start point is invalid
            if '2wp' not in self.nodes[road['start']['id']]:
                last_waypoint = None
            else:
                last_waypoint=self.waypoints[self.nodes[road['start']['id']]['2wp']]
            while s<length:
                p = np.array([start_x, start_y]) - s/length*np.array([start_x-end_x, start_y-end_y])
                # self.nodes[f"new_node_{len(self.nodes)}"]={"x": p[0], "y": p[1], "connected_roads": road["id"]}
                if self.is_point_invalid(p):
                    last_waypoint = None
                    s+=self.waypoints_dis
                    continue
                new_wp=(Waypoints(id=len(self.waypoints), location=p, belong=road["id"]))
                self.waypoints.append(new_wp)
                if last_waypoint is not None:
                    last_waypoint.successor.append(new_wp.id)
                last_waypoint = new_wp
                s+=self.waypoints_dis
            if '2wp' in self.nodes[road['end']['id']] and last_waypoint is not None:
                last_waypoint.successor.append(self.waypoints[self.nodes[road['end']['id']]['2wp']].id)
        # for road in self.map.printable_roads:
        #     self.road2waypoint[road]=len(self.waypoints)
        #     self.waypoints.append(Waypoints(id=len(self.waypoints), location=self.map.get_pos(road, 0.), belong=road))
        # for road in self.map.printable_roads:
        #     last_waypoint=self.waypoints[self.road2waypoint[road]]
        #     s=self.waypoints_dis
        #     for geometry in self.map.printable_roads[road]["geometry"]:
        #         while s<geometry['length']+geometry['s']:
        #             pos = self.map.get_pos(road, s)
        #             new_wp = Waypoints(id=len(self.waypoints), location=pos, belong=road)
        #             self.waypoints.append(new_wp)
        #             last_waypoint.successor.append(new_wp.id)
        #             last_waypoint = new_wp
        #             s+=self.waypoints_dis
        #     for successor in self.map.printable_roads[road]['successor']:
        #         last_waypoint.successor.append(self.road2waypoint[successor])
        for waypoint in self.waypoints:
            for successor in waypoint.successor:
                self.waypoints[successor].predecessor.append(waypoint.id)
        # for low-connected waypoints, search its neighbour
        for idx, waypoint in enumerate(self.waypoints):
            if len(waypoint.successor)+len(waypoint.predecessor)>1: continue
            for jdx, n_wp in enumerate(self.waypoints):
                if jdx==idx:continue
                if np.linalg.norm(np.array(waypoint.location)-np.array(n_wp.location))<self.waypoints_dis:
                    self.waypoints[idx].successor.append(jdx)
                    self.waypoints[jdx].predecessor.append(idx)

    def get_pose(self):
        return self.pose
    
    def reset_covered_length(self):
        self.covered_length=0.

    def set_pose(self, pose):
        self.covered_length+=np.linalg.norm(pose[:2]-self.pose[:2])
        self.pose=pose

    def query_place(self, place_name):
        # one place at one time to simulate time cost
        knowledge = copy.deepcopy(self.place_metadata[place_name])
        knowledge["bounding_box"]=self.building_metadata[knowledge['building']]['bounding_box']
        knowledge_items={place_name: knowledge}
        return knowledge_items
    
    def query_nearby(self, target_pos, threshold=30):
        places_list=[]
        for place in self.place_metadata:
            if is_near_goal(target_pos[0], target_pos[1], self.place_metadata[place]['bounding_box'], self.place_metadata[place]['location'], threshold=threshold):
                places_list.append(place)
        return places_list
    
    def get_nearest_waypoints(self, curr_trans):
        """
        Find and return several nearest waypoint ids from the given curr_trans.
        """
        ret=[]
        start_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2]))
        )
        min_dis2s = np.linalg.norm(np.array(self.waypoints[start_wp_id].location) - np.array(curr_trans[:2]))
        for i in range(len(self.waypoints)):
            if np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2])) <= min_dis2s+self.waypoints_dis:
                ret.append(i)
        return ret
    
    def query_route(self, curr_trans, goal_place):
        """
        Find a route from current pose to the goal_place using waypoint graph.
        
        Args:
            curr_trans (list | np.ndarray): Current [x, y, ...] position of agent
            goal_place (str): Name of the destination place

        Returns:
            list[Waypoints]: Ordered list of waypoints from current pose to goal
        """
        if not self.waypoints:
            raise ValueError("No waypoints available. Call spawn_waypoints() first.")

        # Get goal location
        if goal_place not in self.place_metadata:
            raise ValueError(f"Unknown place: {goal_place}")
        
        goal_pos = self.place_metadata[goal_place]['location'][:2]  # [x, y]
        curr_trans=copy.deepcopy(curr_trans)
        if goal_pos[0]>500 or goal_pos[1]>500:
            goal_pos[0], goal_pos[1]=goal_pos[0]-1000, goal_pos[1]-1000
        if curr_trans[0]>500 or curr_trans[1]>500:
            curr_trans[0], curr_trans[1]=curr_trans[0]-1000, curr_trans[1]-1000

        # 1. Find nearest waypoint to current pose
        start_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2]))
        )
        min_dis2s = np.linalg.norm(np.array(self.waypoints[start_wp_id].location) - np.array(curr_trans[:2]))

        # 2. Find nearest waypoint to goal location
        goal_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(goal_pos))
        )
        min_dis2t = np.linalg.norm(np.array(self.waypoints[goal_wp_id].location) - np.array(goal_pos))

        # 3. Pathfinding: Dijkstra (or BFS if uniform cost) over waypoint graph
        # Using Dijkstra with distance as edge cost
        dist = {i: float('inf') for i in range(len(self.waypoints))}
        prev = {i: None for i in range(len(self.waypoints))}
        heap = []
        for i in range(len(self.waypoints)):
            if np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2])) <= min_dis2s+self.waypoints_dis:
                dist[i] = np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2]))
                heapq.heappush(heap, (dist[i], i))

        while heap:
            d, wp_id = heapq.heappop(heap)
            if d > dist[wp_id]:
                continue
            if wp_id == goal_wp_id:
                break

            current_wp = self.waypoints[wp_id]
            potential = current_wp.successor + current_wp.predecessor
            for succ_id in potential:
                if succ_id >= len(self.waypoints):
                    continue
                succ_wp = self.waypoints[succ_id]
                cost = np.linalg.norm(
                    np.array(current_wp.location) - np.array(succ_wp.location)
                )
                new_dist = dist[wp_id] + cost
                if new_dist < dist[succ_id]:
                    dist[succ_id] = new_dist
                    prev[succ_id] = wp_id
                    heapq.heappush(heap, (new_dist, succ_id))

        # 4. Reconstruct path
        goal_wp_pair=(dist[goal_wp_id]+min_dis2t, goal_wp_id)
        for i in range(len(self.waypoints)):
            if np.linalg.norm(np.array(self.waypoints[i].location) - np.array(goal_pos)) <= min_dis2t+self.waypoints_dis:
                goal_wp_pair=min((dist[i]+np.linalg.norm(np.array(self.waypoints[i].location) - np.array(goal_pos)), i), goal_wp_pair)
        if goal_wp_pair[0] == float('inf'):
            self.logger.error(f"{self.scene_name}: No path found from {curr_trans[:2]} to {goal_place} at {goal_pos}")
            return []
        path = []
        curr = goal_wp_pair[1]
        while curr is not None:
            path.append(list(self.waypoints[curr].location))
            curr = prev[curr]
        path.reverse()

        if not path:
            self.logger.error(f"{self.scene_name}: No valid route found from {curr_trans[:2]} to {goal_place} at {goal_pos}")
            return []
        
        path.append(goal_pos)
        return path
    
    def get_connected_waypoints(self, waypoint_id):
        """
        Find all waypoints connected to the given waypoint_id via successor/predecessor links.
        Performs BFS to collect all reachable waypoints in the graph.

        Returns:
            List[int]: List of waypoint IDs that are connected (including the start).
        """
        if waypoint_id >= len(self.waypoints) or waypoint_id < 0:
            return []

        visited = set()
        queue = [waypoint_id]
        visited.add(waypoint_id)

        while queue:
            current_id = queue.pop(0)
            current_wp = self.waypoints[current_id]

            # Traverse both successors and predecessors for full connectivity
            neighbors = current_wp.successor + current_wp.predecessor
            for neighbor_id in neighbors:
                if neighbor_id < len(self.waypoints) and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

        return sorted(list(visited))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", '-s', type=str, required=True)
    args = parser.parse_args()
    if not os.path.exists(f"ViCo/assets/scenes/{args.scene}/road_data/road_data.pkl"):
        print(f"ViCo/assets/scenes/{args.scene}/road_data/road_data.pkl not exist!")
        exit()
    with open(f'ViCo/assets/scenes/{args.scene}/raw/center.txt', "r") as file:
        for line in file:
            ref_lat, ref_lon = line.strip().split()
        ref_lat, ref_lon = float(ref_lat), float(ref_lon)
    amap=Amap(scene_name=args.scene)
    wps=[wp.location for wp in amap.waypoints]
    xs, ys = zip(*wps)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, 'bo', markersize=3)
    n_wps=amap.get_nearest_waypoints([285.16, -196.96])
    c_wps=set()
    for wp in n_wps:
        c_wps.update(amap.get_connected_waypoints(wp))
    c_wps=[wp.location for wp in amap.waypoints if wp.id in c_wps]
    c_xs, c_ys = zip(*c_wps)
    plt.plot(c_xs, c_ys, 'ro', markersize=3)
    plt.title(f'Amap Waypoints Visualization - {args.scene}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    # plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.show()