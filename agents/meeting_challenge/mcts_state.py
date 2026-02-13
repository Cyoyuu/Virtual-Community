"""
MCTS State representation for meeting challenge agents.
Each state represents the positions of all agents and the current agent's planned destination.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MCTSState:
    """
    Represents a state in the MCTS search tree.
    
    Attributes:
        agent_positions: Dictionary mapping agent names to their [x, y] positions
        current_agent: Name of the agent making the decision
        current_place: Current place name the agent is at or heading to
        depth: Depth of this state in the search tree
    """
    agent_positions: Dict[str, List[float]]
    current_agent: str
    current_place: Optional[str]
    depth: int = 0
    
    def __hash__(self):
        """Make state hashable for use in dictionaries."""
        # Create a hash from agent positions and current place
        pos_tuple = tuple(sorted((name, tuple(pos[:2])) for name, pos in self.agent_positions.items()))
        return hash((pos_tuple, self.current_agent, self.current_place, self.depth))
    
    def __eq__(self, other):
        """Check equality of states."""
        if not isinstance(other, MCTSState):
            return False
        return (self.agent_positions == other.agent_positions and
                self.current_agent == other.current_agent and
                self.current_place == other.current_place and
                self.depth == other.depth)
    
    def calculate_max_distance(self) -> float:
        """
        Calculate the maximum distance between any two agents.
        This is the metric we want to minimize.
        
        Returns:
            Maximum pairwise distance between agents
        """
        if len(self.agent_positions) < 2:
            return 0.0
        
        positions = list(self.agent_positions.values())
        max_dist = 0.0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1 = np.array(positions[i][:2])
                pos2 = np.array(positions[j][:2])
                dist = np.linalg.norm(pos1 - pos2)
                max_dist = max(max_dist, dist)
        
        return max_dist
    
    def calculate_average_distance(self) -> float:
        """
        Calculate the average distance between all pairs of agents.
        
        Returns:
            Average pairwise distance between agents
        """
        if len(self.agent_positions) < 2:
            return 0.0
        
        positions = list(self.agent_positions.values())
        total_dist = 0.0
        count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pos1 = np.array(positions[i][:2])
                pos2 = np.array(positions[j][:2])
                dist = np.linalg.norm(pos1 - pos2)
                total_dist += dist
                count += 1
        
        return total_dist / count if count > 0 else 0.0
    
    def get_reward(self) -> float:
        """
        Calculate reward for this state. Lower distance = higher reward.
        We use negative distance as reward to maximize (minimize distance).
        
        Returns:
            Reward value (negative of max distance)
        """
        max_dist = self.calculate_max_distance()
        # Return negative distance as reward (we want to minimize distance)
        return -max_dist
