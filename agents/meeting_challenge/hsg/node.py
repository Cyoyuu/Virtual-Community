import datetime

class Node:
    def __init__(self, node_id, name, node_type, connectivity_center=None, properties=None, children=None):
        self.node_id = node_id
        self.name = name
        self.type = node_type
        self.connectivity_center = connectivity_center or {"x": 0.0, "y": 0.0}
        self.properties = properties or {}
        self.children = children or []

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "name": self.name,
            "type": self.type,
            "connectivity_center": self.connectivity_center,
            "properties": self.properties,
        }

    @staticmethod
    def from_dict(data):
        return Node(
            node_id=data["node_id"],
            name=data["name"],
            node_type=data["type"],
            connectivity_center=data.get("connectivity_center"),
            properties=data.get("properties", {}),
        )
