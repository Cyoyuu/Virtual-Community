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
import heapq

from ViCo.tools.utils import *

@dataclass
class Waypoints:
    id: int
    location: list[float, float] | None = None
    belong: str | None = None
    predecessor: list = field(default_factory=list)
    successor: list = field(default_factory=list)

class Amap:
    '''walkers only'''
    def __init__(self, map=None, pose=None, place_metadata=None, building_metadata=None):
        self.map=map
        self.pose=pose
        self.covered_length=0.
        self.place_metadata=place_metadata
        self.building_metadata=building_metadata

        self.waypoints = []
        self.road2waypoint = {}
        
    def reset(self, pose):
        self.pose=pose
        self.covered_length=0.

    def spawn_waypoints(self):
        for road in self.map.printable_roads:
            self.road2waypoint[road]=len(self.waypoints)
            self.waypoints.append(Waypoints(id=len(self.waypoints), location=self.map.get_pos(road, 0.), belong=road))
        for road in self.map.printable_roads:
            last_waypoint=self.waypoints[self.road2waypoint[road]]
            s=10.
            for geometry in self.map.printable_roads[road]["geometry"]:
                while s<geometry['length']:
                    pos = self.map.get_pos(road, s)
                    new_wp = Waypoints(id=len(self.waypoints), location=pos, belong=road)
                    self.waypoints.append(new_wp)
                    last_waypoint.successor.append(new_wp.id)
                    last_waypoint = new_wp
                    s+=10
                s=0.
            for successor in self.map.printable_roads[road]['successor']:
                last_waypoint.successor.append(self.road2waypoint[successor])
        for waypoint in self.waypoints:
            for successor in waypoint.successor:
                self.waypoints[successor].predecessor.append(waypoint.id)

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

        # 1. Find nearest waypoint to current pose
        start_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(curr_trans[:2]))
        )

        # 2. Find nearest waypoint to goal location
        goal_wp_id = min(
            range(len(self.waypoints)),
            key=lambda i: np.linalg.norm(np.array(self.waypoints[i].location) - np.array(goal_pos))
        )

        # 3. Pathfinding: Dijkstra (or BFS if uniform cost) over waypoint graph
        # Using Dijkstra with distance as edge cost
        dist = {i: float('inf') for i in range(len(self.waypoints))}
        prev = {i: None for i in range(len(self.waypoints))}
        dist[start_wp_id] = 0
        heap = [(0, start_wp_id)]

        while heap:
            d, wp_id = heapq.heappop(heap)
            if d > dist[wp_id]:
                continue
            if wp_id == goal_wp_id:
                break

            current_wp = self.waypoints[wp_id]
            for succ_id in current_wp.successor:
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
        if dist[goal_wp_id] == float('inf'):
            print(f"No path found from {curr_trans[:2]} to {goal_pos}")
            return []
        path = []
        curr = goal_wp_id
        while curr is not None:
            path.append(self.waypoints[curr])
            curr = prev[curr]
        path.reverse()

        if not path or path[0].id != start_wp_id:
            print(f"No valid route found from {curr_trans[:2]} to {goal_pos}")
            return []
        
        return path

