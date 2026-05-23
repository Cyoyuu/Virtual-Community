#!/usr/bin/env python3
"""
Analyze all refine_history.json files under meeting_challenge/output.
Counts true/false values and computes their ratios, broken down by
sentinel setting (patrol_5, patrol_10, stationary_5, etc.).

Path structure:
  output/{CITY}/{sentinel_type}/{sentinel_setting}/job_{N}/curr_sim/{Person}/refine_history.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def solve_x(ratio: float) -> float:
    """Solve for x given true_ratio = (x + x^2 + x^3) / (3 - x - x^2).
    Rearranges to: x^3 + (1+r)x^2 + (1+r)x - 3r = 0
    Returns the real root in [0, 1], or nan if none found.
    """
    if np.isnan(ratio):
        return float("nan")
    r = ratio
    # coefficients for numpy: highest degree first
    roots = np.roots([1, 1 + r, 1 + r, -3 * r])
    real_roots = [v.real for v in roots if abs(v.imag) < 1e-6 and 0 <= v.real <= 1]
    return float(real_roots[0]) if real_roots else float("nan")


def _counter():
    return {"true": 0, "false": 0, "files": 0}


def print_stats(label: str, counts: dict, indent: int = 0) -> None:
    pad = " " * indent
    total = counts["true"] + counts["false"]
    true_ratio = counts["true"] / total if total > 0 else float("nan")
    false_ratio = counts["false"] / total if total > 0 else float("nan")
    x = solve_x(true_ratio)
    print(
        f"{pad}{label:<22}  files={counts['files']:>4}  total={total:>5}"
        f"  true={counts['true']:>4}  false={counts['false']:>4}"
        f"  true_ratio={true_ratio:.3f}  false_ratio={false_ratio:.3f}  x={x:.4f}"
    )


def analyze_refine_histories(base_dir: str) -> None:
    base_path = Path(base_dir)

    # Explicit traversal matching:
    # output/{CITY}/{agent_dir}/{sentinel_setting}/{job_dir}/curr_sim/{person}/refine_history.json
    files = []
    for city_dir in sorted(base_path.iterdir()):
        if not city_dir.is_dir():
            continue
        print(f"Scanning city: {city_dir.name}")
        for agent_dir in city_dir.iterdir():
            if not (agent_dir.is_dir() and agent_dir.name.startswith("sentinel")):
                continue
            for setting_dir in agent_dir.iterdir():
                if not setting_dir.is_dir():
                    continue
                for job_dir in setting_dir.iterdir():
                    if not job_dir.is_dir():
                        continue
                    curr_sim = job_dir / "curr_sim"
                    if not curr_sim.is_dir():
                        continue
                    for person_dir in curr_sim.iterdir():
                        if not person_dir.is_dir():
                            continue
                        f = person_dir / "refine_history.json"
                        if f.exists():
                            files.append(f)
    files.sort()

    if not files:
        print(f"No refine_history.json files found under {base_dir}")
        return

    # Aggregate buckets
    by_setting = defaultdict(_counter)   # keyed by sentinel_setting (e.g. patrol_10)
    grand = _counter()

    per_file_rows = []

    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  [SKIP] Unexpected format in {filepath}")
            continue

        # Extract sentinel_setting from path:
        #   base_path / CITY / sentinel_type / sentinel_setting / job_N / curr_sim / Person / file
        parts = filepath.relative_to(base_path).parts  # len >= 7
        sentinel_setting = parts[2] if len(parts) >= 7 else "unknown"

        n_true = sum(1 for v in data if v is True)
        n_false = sum(1 for v in data if v is False)

        per_file_rows.append((filepath.relative_to(base_path), n_true, n_false))

        for bucket in (by_setting[sentinel_setting], grand):
            bucket["true"] += n_true
            bucket["false"] += n_false
            bucket["files"] += 1

    # ── Per-file detail ──────────────────────────────────────────────────────
    print("=" * 90)
    print("PER-FILE DETAIL")
    print("=" * 90)
    for rel, n_true, n_false in per_file_rows:
        total = n_true + n_false
        ratio = n_true / total if total > 0 else float("nan")
        print(f"  {rel}")
        print(f"    total={total}  true={n_true}  false={n_false}  true_ratio={ratio:.3f}")

    # ── By sentinel setting ──────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("BY SENTINEL SETTING")
    print("=" * 90)
    for setting in sorted(by_setting):
        print_stats(setting, by_setting[setting], indent=2)

    # ── Grand total ──────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("GRAND TOTAL")
    print("=" * 90)
    print_stats("ALL", grand, indent=2)


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    analyze_refine_histories(str(output_dir))
