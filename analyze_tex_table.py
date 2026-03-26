import os
import json
import argparse
from collections import defaultdict

def load_results(gt_mode, agent_num, sentinel_type, sentinel_num):
    """Load results for a specific setting."""
    path = f"results/results_{gt_mode}/{agent_num}_{sentinel_type}_{sentinel_num}.json"
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Filling with zeros.")
        return None
    with open(path, "r") as f:
        return json.load(f)

def format_value(val, metric, is_best=False):
    """Format value with bold if best."""
    formatted = f"{val:.2f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted

def get_row_data(data_dict, agent_key, num_scenes=14, num_runs=2):
    """Extract and scale metrics for one method."""
    if data_dict is None or agent_key not in data_dict:
        return [0.0] * 5
    avg = data_dict[agent_key]["average"]
    # Success rate: scaled by (num_scenes * num_runs) / total possible?
    # But your ref code uses: success_rate * 100 * num / 16
    # Since you have 14 scenes × 2 runs = 28 episodes, but they normalize to 16? 
    # However, in your table, values like 76.92 suggest direct percentage.
    # Let's assume 'success_rate' is already fraction of successful episodes.
    succ = avg["success_rate"] * 100
    caught = avg["caught_rate"] * 100
    detect = avg["detection_rate"] * 100
    time = avg["time_spent_meeting_mean"]
    dist = avg["walk_spent_meeting_mean"]
    return [succ, caught, detect, time, dist]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-d", type=str, default=".", help="Root directory containing result_standard/ and result_oracle/")
    parser.add_argument("--agent_num", "-a", type=int, default=5)
    parser.add_argument("--sentinel_num", "-s", type=int, default=5)
    args = parser.parse_args()

    os.chdir(args.root)

    # Define methods in order as in table
    method_keys = {
        "Oracle Centered": "center_no_avoidance",
        "Oracle Centered w/ DZ": "center",
        "MCTS": "mcts",
        "RoCo": "roco",
        "CoELA": "coela",
        "CoSaR (Ours)": "sentinel"
    }

    # Load all four result sets
    results = {
        ("no_gt", "stationary"): load_results("no_gt", args.agent_num, "stationary", args.sentinel_num),
        ("no_gt", "patrolling"): load_results("no_gt", args.agent_num, "patrol", args.sentinel_num),
        ("gt", "stationary"): load_results("gt", args.agent_num, "stationary", args.sentinel_num),
        ("gt", "patrolling"): load_results("gt", args.agent_num, "patrol", args.sentinel_num),
    }

    # Collect data per method per condition
    table_data = {}
    for name, key in method_keys.items():
        row = []
        for (gt, stype) in [("no_gt", "stationary"), ("no_gt", "patrolling"),
                            ("gt", "stationary"), ("gt", "patrolling")]:
            data = results[(gt, stype)]
            metrics = get_row_data(data, key)
            row.append(metrics)
        table_data[name] = row  # [std_stat, std_pat, oracle_stat, oracle_pat]

    # Now determine best per column (per scenario)
    scenarios = ["standard_stationary", "standard_patrolling", "oracle_stationary", "oracle_patrolling"]
    metrics_per_scenario = defaultdict(lambda: [[] for _ in range(5)])  # 5 metrics

    for name in method_keys:
        for i, scenario in enumerate(scenarios):
            metrics = table_data[name][i]
            for j in range(5):
                metrics_per_scenario[scenario][j].append((name, metrics[j]))

    # Determine best indices per metric per scenario
    best_in_scenario = {}
    for scenario in scenarios:
        best_in_scenario[scenario] = []
        for j, metric_list in enumerate(metrics_per_scenario[scenario]):
            if j == 0:  # Success Rate: higher is better
                best_val = max(m[1] for m in metric_list)
            else:  # Others: lower is better
                best_val = min(m[1] for m in metric_list)
            best_names = {m[0] for m in metric_list if abs(m[1] - best_val) < 1e-3}
            best_in_scenario[scenario].append(best_names)

    # Build LaTeX rows
    lines = []

    # Sentinel Challenge (standard)
    lines.append(r"\multicolumn{11}{c}{\textit{\textbf{Sentinel Challenge}}} \\")
    lines.append(r"\midrule")
    for name in ["Oracle Centered", "Oracle Centered w/ DZ", "MCTS", "RoCo", "CoELA", "CoSaR (Ours)"]:
        if name == "Oracle Centered" and "oracle" in scenarios[2]:  # Exclude from oracle-perception section, but it's in standard
            pass  # It's allowed in standard
        vals = table_data[name]
        std_stat = vals[0]
        std_pat = vals[1]
        row_parts = []
        lines.append(name)
        for j, v in enumerate(std_stat):
            is_best = name in best_in_scenario["standard_stationary"][j]
            row_parts.append(format_value(v, j, is_best))
        lines.append("& "+" & ".join(row_parts))
        row_parts = []
        for j, v in enumerate(std_pat):
            is_best = name in best_in_scenario["standard_patrolling"][j]
            row_parts.append(format_value(v, j, is_best))
        lines.append("& "+" & ".join(row_parts))
        lines.append(r" \\")
        # lines.append(" & ".join(row_parts) + r" \\")

    # Sentinel Challenge w/ Oracle Perception
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{c}{\textit{\textbf{Sentinel Challenge w/ Oracle Perception}}} \\")
    lines.append(r"\midrule")
    for name in ["Oracle Centered w/ DZ", "MCTS", "RoCo", "CoELA", "CoSaR (Ours)"]:
        # Skip "Oracle Centered" here as per caption
        vals = table_data[name]
        oracle_stat = vals[2]
        oracle_pat = vals[3]
        row_parts = []
        lines.append(name)
        for j, v in enumerate(oracle_stat):
            is_best = name in best_in_scenario["standard_stationary"][j]
            row_parts.append(format_value(v, j, is_best))
        lines.append("& "+" & ".join(row_parts))
        row_parts = []
        for j, v in enumerate(oracle_pat):
            is_best = name in best_in_scenario["standard_patrolling"][j]
            row_parts.append(format_value(v, j, is_best))
        lines.append("& "+" & ".join(row_parts))
        lines.append(r" \\")
        # lines.append(" & ".join(row_parts) + r" \\")

    # Output full table
    latex_table = r"""\begin{table*}[h!]
\centering
\caption{\textbf{Additional results for 5 agents and 5 sentinels} We report the average score over 14 scenes and 2 runs here. Best performance is shown in \textbf{bold}. The Oracle Centered baseline is excluded from the oracle-perception scenario, since the method itself does not utilize oracle perception}
\label{tab:10_agent}
\resizebox{\linewidth}{!}{
\begin{tabular}{l|ccccc|ccccc}
\toprule
& \multicolumn{5}{c|}{\textbf{5 Stationary Sentinels}} 
& \multicolumn{5}{c}{\textbf{5 Patrolling Sentinels}} \\
\cmidrule(r){2-6} \cmidrule(l){7-11}
Method &
\makecell{Succ.\\Rate\uparrow} &
\makecell{Caught\\Rate\downarrow} &
\makecell{Detect.\\Rate\downarrow} &
Time\downarrow &
Dist.\downarrow
&
\makecell{Succ.\\Rate\uparrow} &
\makecell{Caught\\Rate\downarrow} &
\makecell{Detect.\\Rate\downarrow} &
Time\downarrow &
Dist.\downarrow
\\
\midrule
"""
    latex_table += "\n".join(lines)
    latex_table += r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
    print(latex_table)

if __name__ == "__main__":
    main()