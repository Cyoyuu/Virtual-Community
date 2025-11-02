import json
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", "-o", type=str)
args = parser.parse_args()

# === CONFIG ===
json_path = f"{args.output}/results.json"
output_path = f"{args.output}/results_summary.xlsx"

# === Load JSON ===
with open(json_path, "r") as f:
    results = json.load(f)

# Collect all scenes and agent types
scenes = set()
agent_types = []

for agent_type, scene_dict in results.items():
    agent_types.append(agent_type)
    scenes.update(scene_dict.keys())

# Remove 'average' if present
scenes = sorted([s for s in scenes if s.lower() != "average"])

# === Function to build table ===
def build_table(metric_name: str):
    """Build a DataFrame for the given metric (e.g., 'success_rate' or 'caught_rate')."""
    data = {"scene": scenes}
    df = pd.DataFrame(data)

    for agent_type in agent_types:
        col_values = []
        for scene in scenes:
            if scene in results[agent_type]:
                val = results[agent_type][scene].get(metric_name, "")
            else:
                val = ""
            col_values.append(val)
        df[agent_type] = col_values

    return df

# === Build both tables ===
df_success = build_table("success_rate")
df_caught = build_table("caught_rate")

# === Save to Excel (multi-sheet) ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_success.to_excel(writer, sheet_name="success_rate", index=False)
    df_caught.to_excel(writer, sheet_name="caught_rate", index=False)

print(f"✅ Saved multi-sheet Excel to {output_path}")
