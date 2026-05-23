#!/usr/bin/env python3
"""Calculate replanning frequency per episode.

Counts occurrences of "generated response: {'initiate_new_discussion': True"
in non-sentinel agent log files, summed per job (= episode), then averaged
across all jobs.

Usage:
    python meeting_challenge/cal_replanning.py <gt|no_gt> <agent_num> <sentinel_type> <sentinel_num>

Example:
    python meeting_challenge/cal_replanning.py gt 5 stationary 10
        -> reads meeting_challenge/output/*/sentinel_gt_5/stationary_10/*/logs/
        -> writes meeting_challenge/results_gt/5_stationary_10/replanning_frequency.json
"""

import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(ROOT, "output")
NEEDLE = "generated response: {'initiate_new_discussion': True"


def count_in_file(path):
    n = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if NEEDLE in line:
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gt", choices=["gt", "no_gt"])
    parser.add_argument("agent_num", type=int)
    parser.add_argument("sentinel_type")
    parser.add_argument("sentinel_num", type=int)
    args = parser.parse_args()

    method_dir = f"sentinel_{args.gt}_{args.agent_num}"
    scenario_dir = f"{args.sentinel_type}_{args.sentinel_num}"

    job_glob = os.path.join(OUTPUT_BASE, "*", method_dir, scenario_dir, "job_*")
    job_dirs = sorted(glob.glob(job_glob))
    if not job_dirs:
        print(f"No jobs found matching {job_glob}")
        return 1

    per_job = {}
    total_count = 0
    total_jobs = 0
    job_re = re.compile(r".*/output/([^/]+)/[^/]+/[^/]+/(job_\d+)$")

    for job_dir in job_dirs:
        logs_dir = os.path.join(job_dir, "logs")
        if not os.path.isdir(logs_dir):
            continue
        m = job_re.match(job_dir)
        city, job = (m.group(1), m.group(2)) if m else (os.path.basename(os.path.dirname(job_dir)), os.path.basename(job_dir))
        key = f"{city}/{job}"

        agent_counts = {}
        for log_path in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
            name = os.path.splitext(os.path.basename(log_path))[0]
            if name.lower().startswith("sentinel"):
                continue
            agent_counts[name] = count_in_file(log_path)

        job_count = sum(agent_counts.values())
        per_job[key] = {
            "count": job_count,
            "num_non_sentinel_agents": len(agent_counts),
            "per_agent": agent_counts,
        }
        total_count += job_count
        total_jobs += 1

    mean_per_episode = (total_count / total_jobs) if total_jobs else 0.0

    summary = {
        "spec": {
            "gt": args.gt,
            "agent_num": args.agent_num,
            "sentinel_type": args.sentinel_type,
            "sentinel_num": args.sentinel_num,
            "method_dir": method_dir,
            "scenario_dir": scenario_dir,
        },
        "total_jobs": total_jobs,
        "total_replanning_events": total_count,
        "mean_per_episode": mean_per_episode,
        "per_job": per_job,
    }

    out_dir = os.path.join(
        ROOT,
        f"results_{args.gt}",
        f"{args.agent_num}_{args.sentinel_type}_{args.sentinel_num}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "replanning_frequency.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Jobs scanned: {total_jobs}")
    print(f"Total replanning events: {total_count}")
    print(f"Mean per episode: {mean_per_episode:.3f}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
