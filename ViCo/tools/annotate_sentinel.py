import os
import argparse
import json
import shutil
# import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", "-s", type=str)
    parser.add_argument("--x", "-x", type=float)
    parser.add_argument("--y", "-y", type=float)
    args = parser.parse_args()
    sentinel_config_path = f"ViCo/assets/scenes/{args.scene}/agents_num_5/sentinel_config.json"
    if os.path.exists(sentinel_config_path):
        print(f"{sentinel_config_path} already exists")
        exit(0)
    # height_field_path = f"Genesis/genesis/assets/ViCo/..."
    # height_field = utils.load_height_field(height_field_path)
    # height = utils.get_height_at(height_field, args.x, args.y)
    with open(f"ViCo/assets/scenes/{args.scene}/agents_num_5/config.json", "r") as f:
        config = json.load(f)
        height = config['agent_infos'][0]['outdoor_pose'][2]+10
    sentinel_config={}
    sentinel_config['agent_names'] = ["Sentinel 0"]
    sentinel_config['agent_infos'] = [{
            "cash": 1000,
            "held_objects": [
                None,
                None
            ],
            "outdoor_pose": [
                args.x,
                args.y,
                height,
                0.0,
                0.0,
                0.0
            ],
            "current_building": "open space",
            "current_place": None,
            "current_vehicle": None
        }]
    sentinel_config['agent_poses'] = [[
            args.x,
            args.y,
            height,
            0.0,
            0.0,
            0.0
        ]]
    sentinel_config['locator_colors'] = ['cyan']
    sentinel_config['locator_colors_rgb'] = [[
            0.0,
            1.0,
            1.0
        ]]
    sentinel_config['agent_skins'] = ["ViCo/avatars/models/celebrity_Donald_Trump.glb"]
    sentinel_config['patrol_config'] = [{
            "type": "rotating"
        }]
    with open(sentinel_config_path, "w") as f:
        json.dump(sentinel_config, f, indent=4)
    for x in os.listdir(f"ViCo/assets/scenes/{args.scene}/agents_num_5/"):
        if os.path.isdir(f"ViCo/assets/scenes/{args.scene}/agents_num_5/{x}"):
            shutil.copytree(f"ViCo/assets/scenes/{args.scene}/agents_num_5/{x}", f"ViCo/assets/scenes/{args.scene}/agents_num_5/Sentinel 0")
            break