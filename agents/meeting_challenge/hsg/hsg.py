from copy import deepcopy
import uuid
import json

class HSG:
    def __init__(self, id=None, name=None, type=None, connectivity_center=None, properties=None, parent_node=None, nodes=None, edges=None):
        self.id = id or f"hsg_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.type = type
        self.connectivity_center = connectivity_center or {"x": 0.0, "y": 0.0}
        self.properties = properties or {}
        self.parent_node = parent_node
        self.nodes = nodes or {}  # dict[node_id] = Node
        self.edges = edges or []  # list of dicts: {from, to, relation, confidence}

    def add_node(self, node: HSG):
        self.nodes[node.id] = node

    def add_edge(self, from_id, to_id, relation, confidence=1.0):
        edge = {"from": from_id, "to": to_id, "relation": relation, "confidence": confidence}
        if edge not in self.edges:
            self.edges.append(edge)

    def to_node(self):
        node_dict = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "connectivity_center": self.connectivity_center,
            "properties": self.properties,
        }
        return {self.id: node_dict}

    def to_graph(self):
        graph_dict = {
            "nodes": {nid: n.to_node() for nid, n in self.nodes.items()},
            "edges": deepcopy(self.edges),
        }
        return graph_dict
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "connectivity_center": self.connectivity_center,
            "properties": self.properties,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": deepcopy(self.edges),
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def from_dict(data, id):
        nodes = {nid: HSG.from_dict(data, nid) for nid, n in data["nodes"].items()}
        return HSG(
            id=data[id].get("id"),
            name = data[id].get("name"),
            type = data[id].get("type"),
            connectivity_center = data[id].get("connectivity_center", {"x": 0.0, "y": 0.0}),
            properties = data[id].get("properties", {}),
            parent_node=data[id].get("parent_node"),
            nodes=nodes,
            edges=data[id].get("edges", []),
        )
    
    def find_nearest_node(agent_x, agent_y, nodes, threshold=5.0):
        """
        Find which node the agent is currently at or nearest to.

        Args:
            agent_x, agent_y: current coordinates of the agent
            nodes: dict of node_id -> {'x': float, 'y': float}
            threshold: maximum distance (in meters or scene units) to consider "at" a node

        Returns:
            node_id if within threshold, else None
        """
        min_dist = float('inf')
        nearest_node = None

        for node_id, node in nodes.items():
            dx = node.connectivity_center[0] - agent_x
            dy = node.connectivity_center[1] - agent_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id

        if min_dist <= threshold:
            return nearest_node
        else:
            return None
