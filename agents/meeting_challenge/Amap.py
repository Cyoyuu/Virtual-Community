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

from ViCo.tools.utils import *

class Amap:
    def __init__(self, local_map, pose, places_metadata, buildings_metadata):
        self.local_map=local_map
        self.pose=pose
        self.covered_length=0.
        self.places_metadata=places_metadata
        self.buildings_metadata=buildings_metadata
        
    def reset(self, pose):
        self.pose=pose
        self.covered_length=0.

    def get_pose(self):
        return self.pose
    
    def reset_covered_length(self):
        self.covered_length=0.

    def set_pose(self, pose):
        self.covered_length+=np.linalg.norm(pose[:2]-self.pose[:2])
        self.pose=pose

    def query_place(self, place_name):
        # one place at one time to simulate time cost
        knowledge=self.places_metadata[place_name]
        knowledge["bounding_box"]=self.buildings_metadata[knowledge['building']]['bounding_box']
        knowledge_items={place_name: knowledge}
        return knowledge_items
    
    def query_nearby(self, target_pos, threshold=30):
        places_list=[]
        for place in self.places_metadata:
            if is_near_goal(target_pos[0], target_pos[1], self.places_metadata[place]['bounding_box'], self.places_metadata[place]['location'], threshold=threshold):
                places_list.append(place)
        return places_list