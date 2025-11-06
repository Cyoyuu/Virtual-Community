from copy import deepcopy
import uuid
import json
import base64
import io
from PIL import Image
from hsg import HSG
from node import Node

# ------------------------------
# Merge function
# ------------------------------
def merge_subgraph(global_graph: HSG, subgraph: HSG, distance_threshold=15.0):
    """Merge a sub-scene graph into the global scene graph."""

    def _distance(a, b):
        return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5

    def find_match(new_node):
        for existing_node in global_graph.nodes.values():
            name_match = existing_node.name.lower() == new_node.name.lower()
            close_enough = _distance(existing_node.connectivity_center, new_node.connectivity_center) < distance_threshold
            if name_match or close_enough:
                return existing_node
        return None

    id_map = {}
    for sub_id, sub_node in subgraph.nodes.items():
        match = find_match(sub_node)
        if match:
            # merge properties
            for k, v in sub_node.properties.items():
                if k not in match.properties:
                    match.properties[k] = v
            # average connectivity center
            match.connectivity_center = {
                "x": (match.connectivity_center["x"] + sub_node.connectivity_center["x"]) / 2,
                "y": (match.connectivity_center["y"] + sub_node.connectivity_center["y"]) / 2,
            }
            id_map[sub_id] = match.node_id
        else:
            new_id = f"n_{uuid.uuid4().hex[:6]}"
            id_map[sub_id] = new_id
            global_graph.nodes[new_id] = deepcopy(sub_node)
            global_graph.nodes[new_id].node_id = new_id

    for edge in subgraph.edges:
        new_edge = deepcopy(edge)
        new_edge["from"] = id_map.get(edge["from"], edge["from"])
        new_edge["to"] = id_map.get(edge["to"], edge["to"])
        if new_edge not in global_graph.edges:
            global_graph.edges.append(new_edge)

    for sub_id, sub_node in subgraph.nodes.items():
        if sub_node.children:
            global_node_id = id_map[sub_id]
            global_graph.nodes[global_node_id].children = [
                id_map.get(child_id, child_id) for child_id in sub_node.children
            ]
    return global_graph


# ------------------------------
# Image helpers and reconstruction
# ------------------------------
def encode_image(image_path):
    with Image.open(image_path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def reconstruct_subgraph(aerial_img: Image.Image, metadata: dict, generator):
    """Use GPT-4o to reconstruct a local sub-scene graph."""
    system_prompt = (
        "You are a multimodal spatial reasoning model that constructs a hierarchical scene graph (HSG) "
        "from aerial imagery and location metadata. "
        "You must output ONLY a JSON object with two keys: 'nodes' and 'edges'."
    )

    user_prompt = (
        f"Metadata:\n{json.dumps(metadata, indent=2)}\n\n"
        "Analyze the aerial image below and reconstruct a sub-scene graph."
        "Each node must include 'node_id', 'name', 'type', and 'connectivity_center'."
        "Each edge must describe a spatial or semantic relationship.\n"
        "Return only valid JSON with 'nodes' and 'edges'."
    )

    response = generator.generate(f"{system_prompt}\n{user_prompt}", img=aerial_img, json_mode=True)
    subgraph_json = response.parsed if hasattr(response, "parsed") else response.choices[0].message.parsed

    sub_hsg = HSG()
    for node_data in subgraph_json.get("nodes", []):
        node = Node(
            node_id=node_data["id"],
            name=node_data["name"],
            node_type=node_data["type"],
            connectivity_center=node_data.get("connectivity_center", {"x": 0, "y": 0}),
            properties=node_data.get("properties", {}),
            children=node_data.get("children", []),
        )
        sub_hsg.add_node(node)

    for edge in subgraph_json.get("edges", []):
        sub_hsg.add_edge(edge["source"], edge["target"], edge.get("relation", "related"), edge.get("confidence", 1.0))

    return sub_hsg
