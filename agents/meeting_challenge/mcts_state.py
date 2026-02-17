import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import copy


@dataclass
class MCTSState:
    """
    Sequential risk-aware MCTS state.
    """
    agent_positions: Dict[str, List[float]]
    current_agent: str
    current_place: Optional[str]

    time: float
    alive_agents: Dict[str, bool]

    cumulative_distance: float = 0.0
    cumulative_detection: float = 0.0
    depth: int = 0

    def copy(self):
        return MCTSState(
            agent_positions=copy.deepcopy(self.agent_positions),
            current_agent=self.current_agent,
            current_place=self.current_place,
            time=self.time,
            alive_agents=copy.deepcopy(self.alive_agents),
            cumulative_distance=self.cumulative_distance,
            cumulative_detection=self.cumulative_detection,
            depth=self.depth
        )

    def is_terminal(self, max_depth: int, deadline: float) -> bool:
        if self.depth >= max_depth:
            return True
        if self.time >= deadline:
            return True
        if not self.alive_agents.get(self.current_agent, True):
            return True
        return False

    def calculate_max_distance(self) -> float:
        positions = [
            np.array(pos)
            for name, pos in self.agent_positions.items()
            if self.alive_agents.get(name, True)
        ]

        if len(positions) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                max_dist = max(max_dist, d)
        return max_dist

    def get_reward(self, deadline: float) -> float:
        # if not self.alive_agents.get(self.current_agent, True):
        #     return -1000.0

        if self.time >= deadline:
            return -500.0

        max_dist = self.calculate_max_distance()

        reward = 0.0
        reward -= 2.0 * max_dist
        if max_dist <= 20: reward += 1000.0
        reward -= 0.1 * self.cumulative_distance
        reward -= 5.0 * self.cumulative_detection

        return reward
