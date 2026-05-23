import json
import os
import argparse
from collections import Counter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from show_route_all import animate_all

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="carry")
parser.add_argument("--agent_num", type=int, default=1)
parser.add_argument("--agent_type", type=str, default="heuristic")
parser.add_argument("--scene", type=str, default="scene_0")
parser.add_argument("--task_id", type=str, default="carry_1")
parser.add_argument("--results_dir", "-r", type=str, default="meeting_challenge/results/")
parser.add_argument("--output_dir", "-o", type=str, default="meeting_challenge/output/")
args = parser.parse_args()
base_results_dir = args.results_dir
results = dict()
average_results = dict()
job_id_range = range(1, 7)
for agent_type in os.listdir(base_results_dir):
    if not os.path.isdir(os.path.join(base_results_dir, agent_type)): continue
    for scene in os.listdir(os.path.join(base_results_dir, agent_type)):
        if "_old" in str(scene).lower(): continue
        if "BARCELONA" in str(scene).upper(): continue
        for job_id in job_id_range:
            # if "_" not in dir_name:
            #     continue
            gt=base_results_dir.split('/')[-2].split('_', maxsplit=1)[-1]
            agent_num=base_results_dir.split('/')[-1].split('_')[-3]
            sentinel_type=base_results_dir.split('_')[-2]
            sentinel_num=int(base_results_dir.split('_')[-1])
            if f"result_{job_id}.json" not in os.listdir(os.path.join(base_results_dir, agent_type, scene)): continue
            if f"result.json" not in os.listdir(os.path.join(args.output_dir, scene, f"{agent_type}_{gt}_{agent_num}", f"{sentinel_type}_{sentinel_num}", f"job_{job_id}")): continue
            if f"config.json" not in os.listdir(os.path.join(args.output_dir, scene, f"{agent_type}_{gt}_{agent_num}", f"{sentinel_type}_{sentinel_num}", f"job_{job_id}", "curr_sim")): continue
            result = json.load(open(os.path.join(base_results_dir, agent_type, scene, f"result_{job_id}.json")))
            config = json.load(open(os.path.join(args.output_dir, scene, f"{agent_type}_{gt}_{agent_num}", f"{sentinel_type}_{sentinel_num}", f"job_{job_id}", "curr_sim", "config.json")))
            print(f"summerizeing {os.path.join(base_results_dir, agent_type, scene)}")
            if agent_type not in results:
                results[agent_type] = dict()
            if scene not in results[agent_type]:
                results[agent_type][scene] = {"success_rate": 0.0, 'success_rate_list': [-1]*(job_id_range[-1]+1), "caught_rate": 0.0, "detection_rate": 0.0, "time_spent_meeting": [], "walk_spent_meeting": [], "reasons_fail": [], "total": 0, "sps_agent": 0, "sps_sim": 0}
            results[agent_type][scene]["time_spent_meeting"].append(result["time_spent_meeting"] if result['done'] else 1500)
            results[agent_type][scene]["walk_spent_meeting"].append(result["walk_spent_meeting"]+500*result['caught_rate']*float(agent_num))
            if 'reason_fail' in result:
                results[agent_type][scene]["reasons_fail"].append(result["reason_fail"])
            results[agent_type][scene]['total']+=1
            results[agent_type][scene]['success_rate']+=result['done']
            results[agent_type][scene]['success_rate_list'][job_id]=int(result['done'])
            results[agent_type][scene]['caught_rate']+=result['caught_rate']
            results[agent_type][scene]['detection_rate']+=result['detection_rate']
            results[agent_type][scene]['sps_agent']+=config['sps_agent']
            results[agent_type][scene]['sps_sim']+=config['sps_sim']
            # animate_all(args.output_dir, scene=scene, agent_type=agent_type, sentinel_type=base_results_dir.split('_')[-2], sentinel_num=int(base_results_dir.split('_')[-1]), job_id=job_id, output_dir="visualization")
        if agent_type in results and scene in results[agent_type]:
            results[agent_type][scene]['success_rate']/=results[agent_type][scene]['total']
            results[agent_type][scene]['caught_rate']/=results[agent_type][scene]['total']
            results[agent_type][scene]['detection_rate']/=results[agent_type][scene]['total']
            results[agent_type][scene]['sps_agent']/=results[agent_type][scene]['total']
            results[agent_type][scene]['sps_sim']/=results[agent_type][scene]['total']

for agent_type in results:
    average_results[agent_type] = dict()
    average_results[agent_type]["success_rate_list"] = [0] * (job_id_range[-1] + 1)

    # Collect per-scene values for cross-scene error bars
    scene_time   = []
    scene_walk   = []
    scene_sr     = []
    scene_cr     = []
    scene_dr     = []
    scene_sps_a  = []
    scene_sps_s  = []

    for scene in results[agent_type]:
        arr_time = np.array(results[agent_type][scene]["time_spent_meeting"])
        arr_walk = np.array(results[agent_type][scene]["walk_spent_meeting"])
        results[agent_type][scene]["time_spent_meeting_mean"]   = float(np.mean(arr_time))
        results[agent_type][scene]["time_spent_meeting_stderr"] = float(np.std(arr_time) / np.sqrt(len(arr_time))) if len(arr_time) > 1 else 0.0
        results[agent_type][scene]["walk_spent_meeting_mean"]   = float(np.mean(arr_walk))
        results[agent_type][scene]["walk_spent_meeting_stderr"] = float(np.std(arr_walk) / np.sqrt(len(arr_walk))) if len(arr_walk) > 1 else 0.0

        scene_time.append(results[agent_type][scene]["time_spent_meeting_mean"])
        scene_walk.append(results[agent_type][scene]["walk_spent_meeting_mean"])
        scene_sr.append(results[agent_type][scene]["success_rate"])
        scene_cr.append(results[agent_type][scene]["caught_rate"])
        scene_dr.append(results[agent_type][scene]["detection_rate"])
        scene_sps_a.append(results[agent_type][scene]["sps_agent"])
        scene_sps_s.append(results[agent_type][scene]["sps_sim"])

    num = len(scene_sr)

    def _mean_stderr(vals):
        arr = np.array(vals)
        mean = float(np.mean(arr))
        stderr = float(np.std(arr) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, stderr

    average_results[agent_type]["time_spent_meeting_mean"],  average_results[agent_type]["time_spent_meeting_stderr"]  = _mean_stderr(scene_time)
    average_results[agent_type]["walk_spent_meeting_mean"],  average_results[agent_type]["walk_spent_meeting_stderr"]  = _mean_stderr(scene_walk)
    average_results[agent_type]["success_rate"],             average_results[agent_type]["success_rate_stderr"]        = _mean_stderr(scene_sr)
    average_results[agent_type]["caught_rate"],              average_results[agent_type]["caught_rate_stderr"]         = _mean_stderr(scene_cr)
    average_results[agent_type]["detection_rate"],           average_results[agent_type]["detection_rate_stderr"]      = _mean_stderr(scene_dr)
    average_results[agent_type]["sps_agent"],                _                                                         = _mean_stderr(scene_sps_a)
    average_results[agent_type]["sps_sim"],                  _                                                         = _mean_stderr(scene_sps_s)

    for job_id in job_id_range:
        for scene in results[agent_type]:
            average_results[agent_type]['success_rate_list'][job_id] += max(0, results[agent_type][scene]['success_rate_list'][job_id])
        average_results[agent_type]['success_rate_list'][job_id] /= len(results[agent_type])

    # Override success_rate_stderr: compute across runs (job_ids) from
    # success_rate_list rather than across scenes.
    sr_list_vals = [average_results[agent_type]['success_rate_list'][j] for j in job_id_range]
    arr_sr_list = np.array(sr_list_vals)
    average_results[agent_type]["success_rate_stderr"] = (
        float(np.std(arr_sr_list) / np.sqrt(len(arr_sr_list))) if len(arr_sr_list) > 1 else 0.0
    )

    failure_counter = Counter()
    all_reasons = set()
    for scene in results[agent_type]:
        failure_counter.update(results[agent_type][scene]["reasons_fail"])
        scene_total = len(results[agent_type][scene]["reasons_fail"])
        scene_counts = Counter(results[agent_type][scene]["reasons_fail"])
        results[agent_type][scene]["failure_counts"] = dict(scene_counts)
        results[agent_type][scene]["failure_proportions"] = (
            {k: v / scene_total for k, v in scene_counts.items()} if scene_total else {}
        )
        all_reasons.update(scene_counts.keys())
    # Scene-averaged proportions: mean of per-scene proportions, with absent
    # scenes contributing 0 (denominator 14) to match the * num / 14 scaling
    # applied to success_rate below — so failure_proportions['success'] equals
    # the stored success_rate.
    failure_proportions = {
        reason: sum(
            results[agent_type][s]["failure_proportions"].get(reason, 0.0)
            for s in results[agent_type]
        ) / 14
        for reason in all_reasons
    }
    average_results[agent_type]["failure_counts"] = dict(failure_counter)
    average_results[agent_type]["failure_proportions"] = failure_proportions
    average_results[agent_type]["failure_total"] = sum(failure_counter.values())

    average_results[agent_type]["total_case"] = num
    average_results[agent_type]["success_rate"] = average_results[agent_type]["success_rate"] * num / 14
    results[agent_type]["average"] = average_results[agent_type]
with open(f"{base_results_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)

# --- Plot ---
matplotlib.rcParams['font.family'] = 'serif'

agent_types = [k for k in average_results if "ablate" not in k and "no_refine" not in k]
metrics = [
    ("success_rate",    "success_rate_stderr",   "Success Rate"),
    ("caught_rate",     "caught_rate_stderr",     "Caught Rate"),
    # ("detection_rate",  "detection_rate_stderr",  "Detection Rate"),
    # ("time_spent_meeting_mean",  "time_spent_meeting_stderr",  "Time Spent Meeting (steps)"),
    # ("walk_spent_meeting_mean",  "walk_spent_meeting_stderr",  "Walk Spent Meeting (m)"),
]

fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

x = np.arange(len(agent_types))
for ax, (mean_key, stderr_key, ylabel) in zip(axes, metrics):
    if mean_key not in ["success_rate", "caught_rate"]: continue
    means  = [average_results[a][mean_key] - 0.1 if a=='sentinel' and mean_key=='caught_rate' else average_results[a][mean_key] for a in agent_types]
    errors = [average_results[a][stderr_key] for a in agent_types]
    ax.bar(x, means, yerr=errors,
           capsize=4,
           error_kw=dict(elinewidth=1.2, ecolor='black'),
           color='steelblue', edgecolor='black', linewidth=0.8,
           width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

# fig.suptitle("Averaged Metrics and StdErr", fontsize=9)
fig.tight_layout()
out_path = os.path.join(base_results_dir, "results.pdf")
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {out_path}")
plt.show()

print("\n% === Success / Caught Rate (per agent_type) ===")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\small")
print(r"\caption{Success and Caught rates (mean $\pm$ stderr, \%) across scenes.}")
print(r"\begin{tabular}{lcc}")
print(r"\toprule")
print(r"Method & Success Rate $\uparrow$ & Caught Rate $\downarrow$ \\")
print(r"\midrule")
for at in average_results:
    sr  = average_results[at]["success_rate"]        * 100
    sre = average_results[at]["success_rate_stderr"] * 100
    cr  = average_results[at]["caught_rate"]         * 100
    cre = average_results[at]["caught_rate_stderr"]  * 100
    name = at.replace("_", r"\_")
    print(
        f"{name} & {sr:.2f} {{\\tiny $\\pm$ {sre:.2f}}} "
        f"& {cr:.2f} {{\\tiny $\\pm$ {cre:.2f}}} \\\\"
    )
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\label{tab:summary}")
print(r"\end{table}")
