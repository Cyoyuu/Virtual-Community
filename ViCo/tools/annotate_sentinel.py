import os
import argparse
import json
import shutil
# import utils

def annotate_all_rotate(scene, coor_list):
    sentinel_config_path = f"ViCo/assets/scenes/{scene}/agents_num_5/sentinel_config.json"
    # if os.path.exists(sentinel_config_path):
    #     print(f"{sentinel_config_path} already exists")
    #     exit(0)
    # height_field_path = f"Genesis/genesis/assets/ViCo/..."
    # height_field = utils.load_height_field(height_field_path)
    # height = utils.get_height_at(height_field, args.x, args.y)
    sentinel_config={'agent_names': [], 'agent_infos': [], 'agent_poses': [], 'locator_colors': [], 'locator_colors_rgb': [], 'agent_skins': [], 'patrol_config': []}
    for i in range(len(coor_list)):
        x, y = coor_list[i][0], coor_list[i][1]
        with open(f"ViCo/assets/scenes/{scene}/agents_num_5/config.json", "r") as f:
            config = json.load(f)
            height = config['agent_infos'][0]['outdoor_pose'][2]+10
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
                "type": "rotating"
            })
        with open(sentinel_config_path, "w") as f:
            json.dump(sentinel_config, f, indent=4)
        for x in os.listdir(f"ViCo/assets/scenes/{scene}/agents_num_5/"):
            if os.path.exists(f"ViCo/assets/scenes/{scene}/agents_num_5/Sentinel {i}"): continue
            if os.path.isdir(f"ViCo/assets/scenes/{scene}/agents_num_5/{x}"):
                shutil.copytree(f"ViCo/assets/scenes/{scene}/agents_num_5/{x}", f"ViCo/assets/scenes/{scene}/agents_num_5/Sentinel {i}")
                break

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str)
    parser.add_argument("--x", "-x", type=float)
    parser.add_argument("--y", "-y", type=float)
    args = parser.parse_args()

    annotate_all_rotate(args.scene, [[args.x, args.y]])