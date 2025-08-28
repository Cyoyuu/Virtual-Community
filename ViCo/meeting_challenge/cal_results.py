import json
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="carry")
parser.add_argument("--agent_num", type=int, default=1)
parser.add_argument("--agent_type", type=str, default="heuristic")
parser.add_argument("--scene", type=str, default="scene_0")
parser.add_argument("--task_id", type=str, default="carry_1")
parser.add_argument("--output_dir", "-o", type=str, default="ViCo/assistance_challenge/output/")
args = parser.parse_args()
base_output_dir = args.output_dir
results = dict()
average_results = dict()
for scene in os.listdir(base_output_dir):
    if not os.path.isdir(os.path.join(base_output_dir, scene)): continue
    for agent_type in os.listdir(os.path.join(base_output_dir, scene)):
        # if "_" not in dir_name:
        #     continue
        if "result.json" not in os.listdir(os.path.join(base_output_dir, scene, agent_type)): continue
        result = json.load(open(os.path.join(base_output_dir, scene, agent_type, "result.json")))
        print(f"summerizeing {os.path.join(base_output_dir, scene, agent_type)}")
        if agent_type not in results:
            results[agent_type] = dict()
        if scene not in results[agent_type]:
            results[agent_type][scene] = dict()
            for key in result:
                if key != "agent_poses":
                    results[agent_type][scene][key] = result[key]
for agent_type in results:
    average_results[agent_type]=dict()
    average_results[agent_type]["time_spent_meeting"]=0.
    average_results[agent_type]["agent_navigation_time_mean"]=0.
    average_results[agent_type]["agent_navigation_time_stdev"]=0.
    average_results[agent_type]["agent_navigation_length_mean"]=0.
    average_results[agent_type]["agent_navigation_length_stdev"]=0.
    num=0
    for scene in results[agent_type]:
        if "OLD" in scene:continue
        if "agent_navigation_time_mean" not in results[agent_type][scene].keys():continue
        num+=1
        for key in average_results[agent_type]:
            average_results[agent_type][key]+=results[agent_type][scene][key]
    for key in average_results[agent_type]:
        average_results[agent_type][key]/=num
    results[agent_type]["average"]=average_results[agent_type]
with open(f"{base_output_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)
import pdb; pdb.set_trace()