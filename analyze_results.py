import json
import pandas as pd
import os
import argparse
import numpy as np

def generate_one(output_dir, sentinel_num):
    scenes = set()
    for sentinel_type in ['stationary', 'patrol']:
        # === CONFIG ===
        json_path = f"{output_dir}/results_{sentinel_type}_{sentinel_num}.json"
        output_path = f"{output_dir}/results_{sentinel_num}_summary.xlsx"

        # === Load JSON ===
        with open(json_path, "r") as f:
            results[sentinel_type] = json.load(f)

        # Collect all scenes and agent types
        agent_types = ['center_no_avoidance', 'center', 'roco', 'coela', 'sentinel']

        for agent_type, scene_dict in results[sentinel_type].items():
            # agent_types.append(agent_type)
            scenes.update(scene_dict.keys())

    # Remove 'average' if present
    scenes = sorted([s for s in scenes if s.lower() != "average"])

    # === Function to build table ===
    def build_table(metric_name: str):
        """Build a DataFrame for the given metric (e.g., 'success_rate' or 'caught_rate')."""
        data = {"scene": scenes}
        df = pd.DataFrame(data)

        for sentinel_type in ['stationary', 'patrol']:
            for agent_type in agent_types:
                col_values = []
                for scene in scenes:
                    if scene in results[sentinel_type][agent_type]:
                        val = results[sentinel_type][agent_type][scene].get(metric_name, "")
                    else:
                        val = ""
                    col_values.append(val)
                df[f"{sentinel_type}_{agent_type}"] = col_values

        return df

    # === Build both tables ===
    df_success = build_table("success_rate")
    df_caught = build_table("caught_rate")
    df_detection = build_table("detection_rate")

    # === Save to Excel (multi-sheet) ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_success.to_excel(writer, sheet_name="success_rate", index=False)
        df_caught.to_excel(writer, sheet_name="caught_rate", index=False)
        df_detection.to_excel(writer, sheet_name="detection_rate", index=False)

    print(f"✅ Saved multi-sheet Excel to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default='results')
    args = parser.parse_args()
    # === CONFIG ===
    output_path = args.output

    # Define the scenes to group into sections (as in your example)
    # For example, “5 Stationary Sentinels”, “10 Stationary Sentinels”, etc.
    sections = [
        f"{sentinel_num} {sentinel_type} Sentinels" for sentinel_num in [10] for sentinel_type in ['stationary', 'patrolling']
    ]

    # Define row order (agent labels)
    methods = {
        "Center + No avoidance": "center_no_avoidance",
        "Center + Danger Zone avoidance": "center",
        "RoCo + Danger Zone avoidance": "roco",
        "CoELA + Danger Zone avoidance": "coela",
        "Ours": "sentinel"
    }

    # Define metrics to show (column order)
    metrics = ["Success Rate", "Caught Rate", "Detected Rate", "Time Cost", "Distance Traveled"]

    # Helper: safely get a value
    def getv(results, agent_type, scene, key):
        return results.get(agent_type, {}).get(scene, {}).get(key, np.nan)

    # === Construct table ===
    section_tables = []

    for sentinel_num in [5, 10, 20]:
        for sentinel_type in ['stationary', 'patrol']:
            section_name = f"{sentinel_num} {sentinel_type} Sentinels"
            section_data = []
            results=json.load(open(f"{args.output}/results_{sentinel_type}_{sentinel_num}.json", "r"))

            for method_name, method in methods.items():
                # Map method label to agent_type key in JSON
                # You can adjust this mapping according to your JSON structure
                key = method.lower().replace(" ", "_").replace("+", "").replace("-", "")
                if key not in results:
                    key = list(results.keys())[0]  # fallback if missing
                if method not in results: continue

                # Aggregate average metrics over listed scenes
                vals = {m: [] for m in metrics}
                for scene in results[method]:
                    vals["Success Rate"].append(getv(results, key, scene, "success_rate") * 100)
                    vals["Caught Rate"].append(getv(results, key, scene, "caught_rate") * 100)
                    vals["Detected Rate"].append(getv(results, key, scene, "detection_rate") * 100)
                    vals["Time Cost"].append(getv(results, key, scene, "time_spent_meeting_mean"))
                    vals["Distance Traveled"].append(getv(results, key, scene, "walk_spent_meeting_mean"))

                # Take means
                averaged = [np.nanmean(vals[m]) for m in metrics]
                section_data.append([method_name] + averaged)

            df = pd.DataFrame(section_data, columns=["Method"] + metrics)
            df.loc[-1] = ["", "", "", "", "", ""]  # empty row for spacing
            df.index = df.index + 1
            df = df.sort_index()
            section_tables.append((section_name, df))

    # === Write to Excel ===
    # os.makedirs(os.path.dirname(output_path), exist_sok=True)
    with pd.ExcelWriter(os.path.join(output_path, "summary_table.xlsx"), engine="openpyxl") as writer:
        for name, df in section_tables:
            df.to_excel(writer, sheet_name="Summary", startrow=writer.sheets.get("Summary", None).max_row if "Summary" in writer.sheets else 0, index=False, header=True)
            # Add title row manually using ExcelWriter positioning
            worksheet = writer.sheets["Summary"]
            row = worksheet.max_row + 2
            worksheet.cell(row=row, column=1, value=name)
            worksheet.cell(row=row, column=1).font = worksheet.cell(row=row, column=1).font.copy(bold=True)
            # # Write table below title
            # df.to_excel(writer, sheet_name="Summary", startrow=row, index=False)

    print(f"✅ Saved formatted summary table to {output_path}")

    for sentinel_num in [10]:
        generate_one(output_dir=output_path, sentinel_num=sentinel_num)
