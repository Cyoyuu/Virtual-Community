import os
import argparse
import json
import shutil
import random
import pickle
import numpy as np

def _load_obstacle(scene):
    pkl = f"ViCo/assets/scenes/{scene}/obstacle_grid.pkl"
    data = pickle.load(open(pkl, 'rb'))
    grid = data["grid"]
    params = data["parameters"]
    return grid, params

def _point_invalid(grid, params, pt):
    i = int((pt[0] - params["min_x"]) / params["resolution"])
    j = int((pt[1] - params["min_y"]) / params["resolution"])
    if i < 0 or i >= params["nx"] or j < 0 or j >= params["ny"]:
        return True
    return grid[i, j] == 1

def _chebyshev_ok(p, q, lo=20.0, hi=100.0):
    dx = abs(p[0] - q[0])
    dy = abs(p[1] - q[1])
    d = max(dx, dy)
    return (d >= lo) and (d <= hi)

def _sample_valid_pair(grid, params):
    while True:
        x = random.uniform(-300, 300)
        y = random.uniform(-300, 300)
        if _point_invalid(grid, params, (x, y)):
            continue
        rx = random.uniform(-300, 300)
        ry = random.uniform(-300, 300)
        if _point_invalid(grid, params, (rx, ry)):
            continue
        if not _chebyshev_ok((x, y), (rx, ry)):
            continue
        return (x, y), (rx, ry)

def annotate_all_rotate(scene, num_agents):
    sentinel_config_path = f"ViCo/assets/scenes/{scene}/agents_num_5/sentinel_config.json"
    sentinel_config={'agent_names': [], 'agent_infos': [], 'agent_poses': [], 'locator_colors': [], 'locator_colors_rgb': [], 'agent_skins': [], 'patrol_config': []}
    with open(f"ViCo/assets/scenes/{scene}/agents_num_5/config.json", "r") as f:
        config = json.load(f)
        height = config['agent_infos'][0]['outdoor_pose'][2]+10
    grid, params = _load_obstacle(scene)
    for i in range(num_agents):
        (x, y), (rx, ry) = _sample_valid_pair(grid, params)
        sentinel_config['agent_names'].append(f"Sentinel {i}")
        sentinel_config['agent_infos'].append({
                "cash": 1000,
                "held_objects": [
                    None,
                    None
                ],
                "outdoor_pose": [
                    x,
                    y,
                    height,
                    0.0,
                    0.0,
                    0.0
                ],
                "current_building": "open space",
                "current_place": None,
                "current_vehicle": None
            })
        sentinel_config['agent_poses'].append([
                x,
                y,
                height,
                0.0,
                0.0,
                0.0
            ])
        sentinel_config['locator_colors'].append('cyan')
        sentinel_config['locator_colors_rgb'].append([
                0.0,
                1.0,
                1.0
            ])
        sentinel_config['agent_skins'].append("ViCo/avatars/models/celebrity_Donald_Trump.glb")
        sentinel_config['patrol_config'].append({
                "type": "patrolling",
                "route": [
                    [x, y],
                    [rx, ry]
                ],
                "route_index": 0
            })
        for d in os.listdir(f"ViCo/assets/scenes/{scene}/agents_num_5/"):
            if os.path.exists(f"ViCo/assets/scenes/{scene}/agents_num_5/Sentinel {i}"): break
            if os.path.isdir(f"ViCo/assets/scenes/{scene}/agents_num_5/{d}"):
                shutil.copytree(f"ViCo/assets/scenes/{scene}/agents_num_5/{d}", f"ViCo/assets/scenes/{scene}/agents_num_5/Sentinel {i}")
                break
    with open(sentinel_config_path, "w") as f:
        json.dump(sentinel_config, f, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str)
    parser.add_argument("--num", "-n", type=int, default=5)
    args = parser.parse_args()
    annotate_all_rotate(args.scene, args.num)
