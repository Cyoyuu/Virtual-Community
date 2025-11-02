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
job_id_range = range(1, 6)
for agent_type in os.listdir(base_output_dir):
    if not os.path.isdir(os.path.join(base_output_dir, agent_type)): continue
    for scene in os.listdir(os.path.join(base_output_dir, agent_type)):
        if "_old" in str(scene).lower(): continue
        for job_id in job_id_range:
            # if "_" not in dir_name:
            #     continue
            if f"result_{job_id}.json" not in os.listdir(os.path.join(base_output_dir, agent_type, scene)): continue
            result = json.load(open(os.path.join(base_output_dir, agent_type, scene, f"result_{job_id}.json")))
            print(f"summerizeing {os.path.join(base_output_dir, agent_type, scene)}")
            if agent_type not in results:
                results[agent_type] = dict()
            if scene not in results[agent_type]:
                results[agent_type][scene] = {"success_rate": 0.0, "caught_rate": 0.0, "detection_rate": 0.0, "time_spent_meeting": [], "walk_spent_meeting": [], "reasons_fail": [], "total": 0}
            results[agent_type][scene]["time_spent_meeting"].append(result["time_spent_meeting"])
            results[agent_type][scene]["walk_spent_meeting"].append(result["walk_spent_meeting"])
            if 'reason_fail' in result:
                results[agent_type][scene]["reasons_fail"].append(result["reason_fail"])
            results[agent_type][scene]['total']+=1
            results[agent_type][scene]['success_rate']+=result['done']
            results[agent_type][scene]['caught_rate']+=result['caught_rate']
            results[agent_type][scene]['detection_rate']+=result['detection_rate']
        if agent_type in results and scene in results[agent_type]:
            results[agent_type][scene]['success_rate']/=results[agent_type][scene]['total']
            results[agent_type][scene]['caught_rate']/=results[agent_type][scene]['total']
            results[agent_type][scene]['detection_rate']/=results[agent_type][scene]['total']
for agent_type in results:
    average_results[agent_type]=dict()
    average_results[agent_type]["time_spent_meeting_mean"]=0.
    average_results[agent_type]["time_spent_meeting_stderr"]=0.
    average_results[agent_type]["walk_spent_meeting_mean"]=0.
    average_results[agent_type]["walk_spent_meeting_stderr"]=0.
    average_results[agent_type]["success_rate"]=0.
    average_results[agent_type]["caught_rate"]=0.
    average_results[agent_type]["detection_rate"]=0.
    num=0
    for scene in results[agent_type]:
        num+=1
        results[agent_type][scene]["time_spent_meeting_mean"]=np.mean(np.array(results[agent_type][scene]["time_spent_meeting"]))
        results[agent_type][scene]["time_spent_meeting_stderr"]=np.std(np.array(results[agent_type][scene]["time_spent_meeting"]))
        results[agent_type][scene]["walk_spent_meeting_mean"]=np.mean(np.array(results[agent_type][scene]["walk_spent_meeting"]))
        results[agent_type][scene]["walk_spent_meeting_stderr"]=np.std(np.array(results[agent_type][scene]["walk_spent_meeting"]))
        average_results[agent_type]["time_spent_meeting_mean"]+=results[agent_type][scene]["time_spent_meeting_mean"]
        average_results[agent_type]["time_spent_meeting_stderr"]+=results[agent_type][scene]["time_spent_meeting_stderr"]
        average_results[agent_type]["walk_spent_meeting_mean"]+=results[agent_type][scene]["walk_spent_meeting_mean"]
        average_results[agent_type]["walk_spent_meeting_stderr"]+=results[agent_type][scene]["walk_spent_meeting_stderr"]
        average_results[agent_type]["success_rate"]+=results[agent_type][scene]["success_rate"]
        average_results[agent_type]["caught_rate"]+=results[agent_type][scene]["caught_rate"]
        average_results[agent_type]["detection_rate"]+=results[agent_type][scene]["detection_rate"]
    for key in average_results[agent_type]:
        average_results[agent_type][key]/=num
    results[agent_type]["average"]=average_results[agent_type]
with open(f"{base_output_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)
import pdb; pdb.set_trace()