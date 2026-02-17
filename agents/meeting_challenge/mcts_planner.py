import random
import math
import numpy as np
import copy
from typing import Dict, List, Optional
from agents.meeting_challenge.mcts_state import MCTSState


class MCTSNode:
    def __init__(self, state: MCTSState, parent=None, action=None, max_depth=5, logger=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[str] = []
        self.max_depth = max_depth
        self.logger = logger

    def is_terminal(self, deadline):
        return self.state.is_terminal(self.max_depth, deadline)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def select_child(self, exploration_constant):
        best_value = float("-inf")
        best_child = None
        assert self.children
        self.logger.debug("selecting child")

        for child in self.children.values():
            if child.visits == 0:
                ucb = float("inf")
            else:
                exploit = child.value / child.visits
                explore = exploration_constant * math.sqrt(
                    math.log(self.visits) / child.visits
                )
                ucb = exploit + explore
            self.logger.debug(f"ucb:{ucb}, value:{child.value}, visits:{self.visits}")
            if ucb > best_value:
                best_value = ucb
                best_child = child
        assert best_child is not None
        return best_child

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def best_action(self):
        best_action = None
        best_value = float("-inf")

        for action, child in self.children.items():
            if child.visits > 0:
                self.logger.debug(f"action: {action}, value: {child.value}, visits: {child.visits}")
                avg = child.value / child.visits
                if avg > best_value:
                    best_value = avg
                    best_action = action
        return best_action


class MCTSPlanner:
    def __init__(self,
                 agent_name: str,
                 num_simulations=200,
                 max_depth=5,
                 exploration_constant=1.4,
                 speed=1.0,
                 logger=None):

        self.agent_name = agent_name
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.speed = speed
        self.logger = logger

    # -----------------------------------------------------
    # Risk-aware transition
    # -----------------------------------------------------

    def simulate_transition(self,
                            state: MCTSState,
                            action: str,
                            place_locations: Dict[str, List[float]],
                            deadline: float) -> MCTSState:

        new_state = state.copy()

        current_pos = np.array(state.agent_positions[self.agent_name])
        target_pos = np.array(place_locations[action])

        distance = np.linalg.norm(current_pos - target_pos)
        travel_time = distance / self.speed

        new_state.time += travel_time
        new_state.cumulative_distance += distance
        new_state.depth += 1
        new_state.current_place = action

        # ---- Stochastic detection model ----
        detection_prob = min(0.8, distance / 600.0)

        if random.random() < detection_prob:
            new_state.cumulative_detection += 1

            # Capture probability
            if random.random() < 0.2:
                new_state.alive_agents[self.agent_name] = False

        new_state.agent_positions[self.agent_name] = target_pos.tolist()

        # assuming other agents are going to the same place
        for agent_name in new_state.agent_positions:
            dist_to_target = np.linalg.norm(np.array(state.agent_positions[agent_name]) - target_pos)
            direction = (target_pos - np.array(state.agent_positions[agent_name])) / dist_to_target
            dist_to_cover = np.linalg.norm(direction * min(self.speed * travel_time, dist_to_target))
            new_pos = np.array(state.agent_positions[agent_name]) + direction * min(self.speed * travel_time, dist_to_target)
            new_state.cumulative_distance += dist_to_cover
            detection_prob = min(0.8, dist_to_cover / 600.0)
            if random.random() < detection_prob:
                new_state.cumulative_detection += 1
                if random.random() < 0.2:
                    new_state.alive_agents[agent_name] = False
            new_state.agent_positions[agent_name] = new_pos.tolist()

        return new_state

    # -----------------------------------------------------
    # Rollout
    # -----------------------------------------------------

    def rollout(self,
                state: MCTSState,
                place_locations,
                candidate_places,
                deadline):

        current = state

        while not current.is_terminal(self.max_depth, deadline):
            action = random.choice(candidate_places)
            current = self.simulate_transition(
                current, action, place_locations, deadline
            )

        return current.get_reward(deadline)

    # -----------------------------------------------------
    # Candidate filtering
    # -----------------------------------------------------

    def filter_places(self,
                      agent_positions,
                      place_locations,
                      k=10):

        positions = list(agent_positions.values())
        center = np.mean(np.array(positions), axis=0)

        scored = []
        for place, pos in place_locations.items():
            dist = np.linalg.norm(np.array(pos) - center)
            scored.append((dist, place))

        scored.sort()
        return [p for _, p in scored[:k]]

    # -----------------------------------------------------
    # Main planning
    # -----------------------------------------------------

    def plan(self,
             agent_positions: Dict[str, List[float]],
             place_locations: Dict[str, List[float]],
             current_time_seconds: float,
             deadline_seconds: float) -> Optional[str]:

        if not place_locations:
            return None

        candidate_places = self.filter_places(
            agent_positions, place_locations
        )

        root_state = MCTSState(
            agent_positions=copy.deepcopy(agent_positions),
            current_agent=self.agent_name,
            current_place=None,
            time=current_time_seconds,
            alive_agents={name: True for name in agent_positions},
            cumulative_distance=0.0,
            cumulative_detection=0.0,
            depth=0
        )

        root = MCTSNode(root_state, max_depth=self.max_depth, logger=self.logger)
        root.untried_actions = candidate_places.copy()

        import tqdm
        for _ in tqdm.tqdm(range(self.num_simulations)):

            node = root
            if node is None:
                self.logger.warning(f"simulation {_}, root is None.")
            else:
                self.logger.debug(f"simulation {_}, root is {root}")
            assert node is not None

            # Selection
            while (not node.is_terminal(deadline_seconds) and
                   node.is_fully_expanded() and
                   node.children):
                node = node.select_child(self.exploration_constant)

            # Expansion
            if (not node.is_terminal(deadline_seconds) and
                    node.untried_actions):

                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)

                new_state = self.simulate_transition(
                    node.state,
                    action,
                    place_locations,
                    deadline_seconds
                )

                child = MCTSNode(new_state,
                                 parent=node,
                                 action=action,
                                 max_depth=self.max_depth,
                                 logger=self.logger)

                child.untried_actions = candidate_places.copy()
                node.children[action] = child
                node = child

            # Simulation
            reward = self.rollout(
                node.state,
                place_locations,
                candidate_places,
                deadline_seconds
            )
            self.logger.debug(f"rollout: reward is {reward}")

            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent

        return root.best_action()
