"""
MCTS-based agent for meeting challenge.
Uses Monte Carlo Tree Search to plan the next place to visit to minimize distance between agents.
"""
import ast
import os
import random
import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from agents.agent import Agent
from agents.memory import SemanticMemory, EventInstance
from agents.meeting_challenge.base_nav import *
from agents.meeting_challenge.mcts_state import MCTSState
from modules.Amap import Route, RouteNode
from tools.utils import *
from tools.model_manager import global_model_manager


class MCTSNode:
    """
    Node in the MCTS search tree.
    """
    def __init__(self, state: MCTSState, parent=None, action=None, max_depth=3):
        self.state = state
        self.parent = parent
        self.action = action  # The action (place name) that led to this node
        self.children: Dict[str, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0  # Total value accumulated
        self.untried_actions: List[str] = []
        self.max_depth = max_depth
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (max depth reached)."""
        return self.state.depth >= self.max_depth
    
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """
        Select child using UCB1 formula.
        
        Args:
            exploration_constant: Exploration constant (default sqrt(2))
        
        Returns:
            Selected child node
        """
        best_value = float('-inf')
        best_child = None
        
        for action, child in self.children.items():
            if child.visits == 0:
                ucb_value = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        
        return best_child
    
    
    def update(self, reward: float):
        """
        Update node statistics after simulation.
        
        Args:
            reward: Reward from simulation
        """
        self.visits += 1
        self.value += reward
    
    def get_best_action(self) -> Optional[str]:
        """
        Get the action with the highest average value.
        
        Returns:
            Best action (place name) or None if no children
        """
        if not self.children:
            return None
        
        best_action = None
        best_value = float('-inf')
        
        for action, child in self.children.items():
            if child.visits > 0:
                avg_value = child.value / child.visits
                if avg_value > best_value:
                    best_value = avg_value
                    best_action = action
        
        return best_action


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for selecting the next place to visit.
    """
    def __init__(self, agent_name: str, num_simulations: int = 100, max_depth: int = 3, 
                 exploration_constant: float = 1.414, logger=None):
        """
        Initialize MCTS planner.
        
        Args:
            agent_name: Name of the agent using this planner
            num_simulations: Number of MCTS simulations per planning step
            max_depth: Maximum depth of search tree
            exploration_constant: UCB1 exploration constant
            logger: Logger instance
        """
        self.agent_name = agent_name
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.logger = logger
        self.places_cache: List[str] = []
        self.place_locations_cache: Dict[str, List[float]] = {}
    
    def update_places(self, places: List[str], place_locations: Dict[str, List[float]]):
        """
        Update the list of available places and their locations.
        
        Args:
            places: List of place names
            place_locations: Dictionary mapping place names to [x, y] locations
        """
        self.places_cache = places
        self.place_locations_cache = place_locations
    
    def simulate_state(self, state: MCTSState, place_locations: Dict[str, List[float]], 
                       available_places: List[str]) -> MCTSState:
        """
        Simulate a random action from the current state.
        This is used during the simulation phase of MCTS.
        
        Args:
            state: Current state
            place_locations: Dictionary mapping place names to locations
            available_places: List of available place names
        
        Returns:
            New state after random action
        """
        if not available_places or state.depth >= self.max_depth:
            return state
        
        # Randomly select a place to go to
        next_place = random.choice(available_places)
        
        # Create new state with agent moved to that place
        new_positions = copy.deepcopy(state.agent_positions)
        if next_place in place_locations:
            new_positions[state.current_agent] = place_locations[next_place][:2]
        
        return MCTSState(
            agent_positions=new_positions,
            current_agent=state.current_agent,
            current_place=next_place,
            depth=state.depth + 1
        )
    
    def rollout(self, state: MCTSState, place_locations: Dict[str, List[float]], 
                available_places: List[str]) -> float:
        """
        Perform a random rollout from the given state to estimate its value.
        
        Args:
            state: Starting state
            place_locations: Dictionary mapping place names to locations
            available_places: List of available place names
        
        Returns:
            Estimated reward from rollout
        """
        current_state = state
        depth = state.depth
        
        # Perform random actions until max depth
        while depth < self.max_depth and available_places:
            current_state = self.simulate_state(current_state, place_locations, available_places)
            depth += 1
        
        # Return reward from final state
        return current_state.get_reward()
    
    def plan(self, agent_positions: Dict[str, List[float]], 
             available_places: List[str],
             place_locations: Dict[str, List[float]]) -> Optional[str]:
        """
        Plan the next place to visit using MCTS.
        
        Args:
            agent_positions: Dictionary mapping agent names to their [x, y] positions
            available_places: List of available place names
            place_locations: Dictionary mapping place names to [x, y] locations
        
        Returns:
            Best place name to visit, or None if no valid action
        """
        if not available_places:
            return None
        
        # Create root state
        root_state = MCTSState(
            agent_positions=copy.deepcopy(agent_positions),
            current_agent=self.agent_name,
            current_place=None,
            depth=0
        )
        root_node = MCTSNode(root_state, max_depth=self.max_depth)
        root_node.untried_actions = available_places.copy()
        
        # Perform MCTS simulations
        for _ in range(self.num_simulations):
            # Selection: traverse from root to leaf
            node = root_node
            while node.children and node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(self.exploration_constant)
            
            # Expansion: if not terminal and has untried actions
            if not node.is_terminal() and not node.is_fully_expanded():
                action = random.choice(node.untried_actions)
                
                # Create new state after taking action
                new_positions = copy.deepcopy(node.state.agent_positions)
                if action in place_locations:
                    new_positions[node.state.current_agent] = place_locations[action][:2]
                
                new_state = MCTSState(
                    agent_positions=new_positions,
                    current_agent=node.state.current_agent,
                    current_place=action,
                    depth=node.state.depth + 1
                )
                
                child = MCTSNode(new_state, parent=node, action=action, max_depth=node.max_depth)
                child.untried_actions = available_places.copy()
                node.children[action] = child
                if action in node.untried_actions:
                    node.untried_actions.remove(action)
                node = child
            
            # Simulation: random rollout
            reward = self.rollout(node.state, place_locations, available_places)
            
            # Backpropagation: update values up the tree
            while node is not None:
                node.update(reward)
                node = node.parent
        
        # Return best action
        best_action = root_node.get_best_action()
        if best_action is None and available_places:
            # Fallback: return place closest to center of all agents
            return self._get_center_place(agent_positions, available_places, place_locations)
        
        return best_action
    
    def _get_center_place(self, agent_positions: Dict[str, List[float]], 
                          available_places: List[str],
                          place_locations: Dict[str, List[float]]) -> Optional[str]:
        """
        Fallback: get place closest to the center of all agents.
        
        Args:
            agent_positions: Dictionary mapping agent names to positions
            available_places: List of available place names
            place_locations: Dictionary mapping place names to locations
        
        Returns:
            Place name closest to center, or None
        """
        if not agent_positions or not available_places:
            return None
        
        # Calculate center of all agents
        positions = list(agent_positions.values())
        center = np.mean([np.array(pos[:2]) for pos in positions], axis=0)
        
        # Find closest place to center
        best_place = None
        min_dist = float('inf')
        
        for place in available_places:
            if place in place_locations:
                place_pos = np.array(place_locations[place][:2])
                dist = np.linalg.norm(place_pos - center)
                if dist < min_dist:
                    min_dist = dist
                    best_place = place
        
        return best_place


class MCTSMeetingAgent(BaseNavigationMeetingAgent):
    """
    Agent that uses MCTS to plan navigation to minimize distance between agents.
    """
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False, ablate="",
                 mcts_simulations=100, mcts_max_depth=3, mcts_exploration=1.414):
        """
        Initialize MCTS agent.
        
        Args:
            name: Agent name
            pose: Initial pose
            info: Agent info
            sim_path: Simulation path
            no_react: Whether to disable reactions
            debug: Debug mode
            logger: Logger instance
            lm_source: Language model source
            lm_id: Language model ID
            max_tokens: Max tokens for generation
            temperature: Temperature for generation
            top_p: Top-p for generation
            init_generator: Whether to initialize generator
            detect_interval: Detection interval
            num_agents: Number of agents
            enable_danger_zone: Enable danger zone detection
            ablate: Ablation string
            mcts_simulations: Number of MCTS simulations
            mcts_max_depth: Maximum depth for MCTS
            mcts_exploration: Exploration constant for MCTS
        """
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, 
                        max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, 
                        enable_danger_zone, ablate)
        
        # Initialize MCTS planner
        self.mcts_planner = MCTSPlanner(
            agent_name=name,
            num_simulations=mcts_simulations,
            max_depth=mcts_max_depth,
            exploration_constant=mcts_exploration,
            logger=logger
        )
        
        # Planning state
        self.planned_place = None
        self.planning_interval = 50  # Replan every N steps
    
    def reset(self, name, pose):
        """Reset agent state."""
        super().reset(name, pose)
        self.planned_place = None
    
    def _process_obs(self, obs):
        """Process observations."""
        super()._process_obs(obs)
        self.process_obs_with_sptial_knowledge(obs)
    
    def _get_available_places(self) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        Get list of available places and their locations.
        
        Returns:
            Tuple of (list of place names, dictionary mapping place names to locations)
        """
        places = self.s_mem.get_places()
        place_locations = {}
        
        for place in places:
            place_dict = self.s_mem.get_knowledge(place)
            if place_dict is None:
                continue
            
            location = np.array([place_dict["location"][0], place_dict["location"][1]])
            if place_dict["building"] != "open space":
                location[0], location[1] = location[0] - 1000, location[1] - 1000
            
            if location[0] > 500:  # Hack for coordinate system
                location[0], location[1] = location[0] - 1000, location[1] - 1000
            
            place_locations[place] = location[:2].tolist()
        
        return places, place_locations
    
    def _get_agent_positions(self) -> Dict[str, List[float]]:
        """
        Get current positions of all agents.
        
        Returns:
            Dictionary mapping agent names to [x, y] positions
        """
        agent_positions = {}
        
        for agent_name, agent_info in self.obs['agent_pos_dict'].items():
            pose = agent_info['pose'][:2]
            
            # Adjust coordinates if needed
            if agent_info.get('place') is not None:
                pose = [pose[0] - 1000, pose[1] - 1000]
            
            if pose[0] > 500:  # Hack for coordinate system
                pose = [pose[0] - 1000, pose[1] - 1000]
            
            agent_positions[agent_name] = pose
        
        return agent_positions
    
    def _plan_next_place(self) -> Optional[str]:
        """
        Use MCTS to plan the next place to visit.
        
        Returns:
            Place name to visit, or None
        """
        # Get available places and their locations
        available_places, place_locations = self._get_available_places()
        
        if not available_places:
            return None
        
        # Get current agent positions
        agent_positions = self._get_agent_positions()
        
        # Update MCTS planner with current places
        self.mcts_planner.update_places(available_places, place_locations)
        
        # Plan using MCTS
        next_place = self.mcts_planner.plan(
            agent_positions=agent_positions,
            available_places=available_places,
            place_locations=place_locations
        )
        
        if self.logger:
            self.logger.info(f"{self.name} MCTS planned next place: {next_place}")
            if agent_positions:
                max_dist = MCTSState(
                    agent_positions=agent_positions,
                    current_agent=self.name,
                    current_place=None
                ).calculate_max_distance()
                self.logger.info(f"Current max distance between agents: {max_dist:.2f}")
        
        return next_place
    
    def _act(self, obs):
        """
        Main action selection using MCTS planning.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Action dictionary
        """
        if self.banned:
            if self.pose[0] > -1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        
        # Replan periodically or if no current plan
        if self.planned_place is None or self.mode_time_counter % self.planning_interval == 0:
            self.planned_place = self._plan_next_place()
            
            if self.planned_place is None:
                return {"type": "wait"}
            
            # Enter navigation mode
            self.enter_navigation_mode(goal_place=self.planned_place)
        
        # Navigate to planned place
        action, arrived = self.city_navigate(self.planned_place)
        
        if arrived:
            # Check if all agents are at the same location
            agent_positions = self._get_agent_positions()
            if len(agent_positions) > 1:
                # Check if we're close enough to other agents
                my_pos = np.array(agent_positions.get(self.name, [0, 0])[:2])
                all_close = True
                for agent_name, agent_pos in agent_positions.items():
                    if agent_name != self.name:
                        dist = np.linalg.norm(my_pos - np.array(agent_pos[:2]))
                        if dist > 10.0:  # Threshold for "meeting"
                            all_close = False
                            break
                
                if all_close:
                    action = {'type': 'task_complete'}
                else:
                    # Replan to get closer
                    self.planned_place = None
                    return self._act(obs)
            else:
                action = {'type': 'task_complete'}
        
        self.mode_time_counter += 1
        self.last_action = action
        return action
