import os
import sys
import genesis as gs
import psutil
import json
import numpy as np
import pathlib
current_directory = os.getcwd()
sys.path.insert(0, current_directory)
from ViCo.modules.indoor_scenes.usd_scene import place_usd_scene_with_ratio
from ViCo.modules.indoor_scenes.architect_scene import load_indoor_scene


def load_default_room(env, place_name='default_room'):
    indoor_scene = "ViCo/modules/indoor_scenes/scenes/dormitory-8.json"  # a default room
    
    with open(indoor_scene, 'r') as f:
        indoor_scene_place = json.load(f)

    active_places_info = indoor_scene_place
    try:
        place_cameras = load_indoor_scene(env,
                                            place=[indoor_scene_place],
                                            offset_x=1500,
                                            offset_y=1500,
                                            offset_z=-50,
                                            no_objects=True)[0][1]
    except Exception as e:
        gs.logger.error(f"Error loading indoor scene for place {place_name}: {e}")
    return active_places_info, place_cameras

def load_indoor_room(env, place, place_name, load_indoor_objects):

    indoor_scene = place['scene']

    scene_not_found = False
    if indoor_scene.endswith('.json'):
        scene_not_found = not os.path.exists(indoor_scene)
    else:
        scene_not_found = not os.path.exists(f"Genesis/genesis/assets/ViCo/scene/commercial_scenes/scenes/{indoor_scene}_usd")
    if scene_not_found:
        gs.logger.error(f"Scene for place {place_name} is not found.")
        return None, None

    if indoor_scene.endswith('.json'):
        with open(indoor_scene, 'r') as f:
            indoor_scene_place = json.load(f)

        active_places_info = indoor_scene_place
        try:
            load_objects = load_indoor_objects
            
            place_cameras = load_indoor_scene(env,
                                                place=[indoor_scene_place],
                                                offset_x=place['location'][0],
                                                offset_y=place['location'][1],
                                                offset_z=place['location'][2],
                                                no_objects=not load_objects)[0][1]
        except Exception as e:
            gs.logger.error(f"Error loading indoor scene for place {place_name}: {e}")
    else:
        # usd assets
        with pathlib.Path(f"./ViCo/modules/indoor_scenes/scenes/{indoor_scene}.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        offset = np.array(place['location'])
        cameras = data.get("cameras", [])
        place_cameras = []
        for cam in cameras:
            pos = cam['pos'] + offset
            lookat = cam['lookat'] + offset
            place_cameras.append(
                (pos, lookat)
            )

        active_places_info = {
            'init_avatar_poses': data.get("avatar_pos", [])
        }

        usd_file = f"Genesis/genesis/assets/ViCo/scene/commercial_scenes/scenes/{indoor_scene}_usd/start_result_raw.usd"

        load_objects = load_indoor_objects
        place_usd_scene_with_ratio(usd_file,
                        env,
                        global_pos=offset,
                        load_objects=load_objects
        )
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        gs.logger.info(f"Memory used: {mem} MB")
    return active_places_info, place_cameras