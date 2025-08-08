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
    def __init__(self, local_map):
        self.local_map=local_map
    def get_pose(self, curr_pos):
        return None