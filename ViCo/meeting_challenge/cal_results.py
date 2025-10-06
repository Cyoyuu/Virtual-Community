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
parser.add_argument("--output_dir", "-o", type=str, default="ViCo/meeting_challenge/results/")
args = parser.parse_args()
base_output_dir = args.output_dir
results = dict()
average_results = dict()
for agent_type in os.listdir(base_output_dir):
    if not os.path.isdir(os.path.join(base_output_dir, agent_type)): continue
    for scene in os.listdir(os.path.join(base_output_dir, agent_type)):
        if "_old" in str(scene).lower(): continue
        for job_id in range(5):
            # if "_" not in dir_name:
            #     continue
            if f"result_{job_id}.json" not in os.listdir(os.path.join(base_output_dir, agent_type, scene)): continue
            result = json.load(open(os.path.join(base_output_dir, agent_type, scene, f"result_{job_id}.json")))
            print(f"summerizeing {os.path.join(base_output_dir, agent_type, scene)}")
            if agent_type not in results:
                results[agent_type] = dict()
            if scene not in results[agent_type]:
                results[agent_type][scene] = {"success": 0, "time_spent_meeting": [], "total": 0}
            results[agent_type][scene]["time_spent_meeting"].append(result["time_spent_meeting"])
            results[agent_type][scene]['total']+=1
            if result['done']:
                results[agent_type][scene]['success']+=1
            # for key in result:
            #     if key != "agent_poses":
            #         results[agent_type][scene][key] = result[key]
            # for key in ['time', 'length']:
            #     results[agent_type][scene][f"agent_navigation_{key}_mean"]=np.mean(np.array(results[agent_type][scene][f"agent_navigation_{key}"]))
            #     results[agent_type][scene][f"agent_navigation_{key}_stdev"]=np.std(np.array(results[agent_type][scene][f"agent_navigation_{key}"]))
for agent_type in results:
    average_results[agent_type]=dict()
    average_results[agent_type]["time_spent_meeting_mean"]=0.
    average_results[agent_type]["time_spent_meeting_stderr"]=0.
    average_results[agent_type]["success_rate"]=0.
    num=0
    for scene in results[agent_type]:
        num+=1
        results[agent_type][scene]["time_spent_meeting_mean"]=np.mean(np.array(results[agent_type][scene]["time_spent_meeting"]))
        results[agent_type][scene]["time_spent_meeting_stderr"]=np.std(np.array(results[agent_type][scene]["time_spent_meeting"]))
        average_results[agent_type]["time_spent_meeting_mean"]+=results[agent_type][scene]["time_spent_meeting_mean"]
        average_results[agent_type]["time_spent_meeting_stderr"]+=results[agent_type][scene]["time_spent_meeting_stderr"]
        average_results[agent_type]["success_rate"]+=results[agent_type][scene]["success"]/results[agent_type][scene]["total"]
    for key in average_results[agent_type]:
        average_results[agent_type][key]/=num
    results[agent_type]["average"]=average_results[agent_type]
with open(f"{base_output_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)
import pdb; pdb.set_trace()