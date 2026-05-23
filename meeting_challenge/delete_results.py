#!/usr/bin/env python3
"""Batch delete result.json files in meeting_challenge/output/<CITY>/...

Spec format (all lowercase, one per line; empty line ends input):
    gt 5 stationary 10        # header: <gt|no_gt> <level> <scenario> <num>
    roco job 5                # entry:  <method> [job] <n> [<n> ...]
    roco job 3 4              # multiple jobs on one line are OK
    cosar job 4               # alias:  cosar -> sentinel
    sentinel job 1

For every entry the script expands the glob

    meeting_challenge/output/*/<method>_<gt_setting>/<scenario>/job_<n>/result.json

across all cities, prints the matched files, and asks for y/n confirmation
before deleting.

Usage:
    python meeting_challenge/delete_results.py
    python meeting_challenge/delete_results.py < spec.txt
    python meeting_challenge/delete_results.py -y < spec.txt   # skip confirmation
"""

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(ROOT, "output")

METHOD_ALIASES = {
    "cosar": "sentinel",
}


def parse_header(parts):
    i = 0
    if parts[i] == "no" and parts[i + 1] == "gt":
        gt_setting = f"no_gt_{parts[i + 2]}"
        i += 3
    elif parts[i] == "no_gt":
        gt_setting = f"no_gt_{parts[i + 1]}"
        i += 2
    elif parts[i] == "gt":
        gt_setting = f"gt_{parts[i + 1]}"
        i += 2
    else:
        raise ValueError(f"bad header: {' '.join(parts)}")
    scenario = f"{parts[i]}_{parts[i + 1]}"
    return gt_setting, scenario


def parse_entry(parts):
    """Parse '<method> [job] <n> [<n> ...]' into (method, [job_n, ...])."""
    method = METHOD_ALIASES.get(parts[0], parts[0])
    rest = parts[1:]
    if rest and rest[0] == "job":
        rest = rest[1:]
    job_ns = [p for p in rest if p]
    if not job_ns or not all(p.isdigit() for p in job_ns):
        raise ValueError(f"bad entry: {' '.join(parts)}")
    return method, job_ns


def read_spec_and_answer():
    """Read spec lines and (for piped input) an optional confirmation answer.

    Interactive: empty line ends the spec; confirmation is read later via input().
    Piped: lines before the first empty line are the spec; the first non-empty
    line after the empty line, if any, is the confirmation answer.
    """
    interactive = sys.stdin.isatty()
    if interactive:
        print("Enter spec (empty line to finish):")
        print("  Header: gt <level> <scenario> <num>     (or: no_gt <level> ...)")
        print("  Entries: <method> job <n>")
        print()

    spec, answer = [], None
    saw_blank = False
    for raw in sys.stdin:
        line = raw.strip().lower()
        if not line:
            if interactive:
                break
            saw_blank = True
            continue
        if saw_blank:
            answer = line
            break
        spec.append(line)
    return spec, answer


def confirm(prompt, piped_answer):
    if piped_answer is not None:
        print(f"{prompt}{piped_answer}")
        return piped_answer == "y"
    if sys.stdin.isatty():
        return input(prompt).strip().lower() == "y"
    try:
        with open("/dev/tty", "r") as tty:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            return tty.readline().strip().lower() == "y"
    except OSError:
        print("\nNo TTY available for confirmation; pass -y or include a y/n line in piped input.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch delete result.json files.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    lines, piped_answer = read_spec_and_answer()
    if args.yes:
        piped_answer = "y"
    if len(lines) < 2:
        print("Need a header line and at least one entry.")
        return 1

    try:
        gt_setting, scenario = parse_header(lines[0].split())
    except (ValueError, IndexError) as e:
        print(f"Failed to parse header: {e}")
        return 1

    entries = []
    for line in lines[1:]:
        try:
            entries.append(parse_entry(line.split()))
        except (ValueError, IndexError) as e:
            print(f"Skipping '{line}': {e}")

    if not entries:
        print("No valid entries.")
        return 1

    files = []
    print(f"\nResolving entries under {gt_setting}/{scenario}:")
    for method, job_ns in entries:
        for job_n in job_ns:
            pattern = os.path.join(
                OUTPUT_BASE, "*", f"{method}_{gt_setting}", scenario, f"job_{job_n}", "result.json"
            )
            matches = sorted(glob.glob(pattern))
            if matches:
                print(f"  {method} job_{job_n}: {len(matches)} match(es)")
            else:
                print(f"  {method} job_{job_n}: no matches")
            files.extend(matches)

    if not files:
        print("\nNothing to delete.")
        return 0

    print(f"\nFiles to delete ({len(files)}):")
    for f in files:
        print(f"  {f}")

    if not confirm(f"\nDelete these {len(files)} files? [y/N]: ", piped_answer):
        print("Aborted.")
        return 0

    deleted = 0
    for f in files:
        try:
            os.remove(f)
            deleted += 1
        except OSError as e:
            print(f"Failed to delete {f}: {e}")
    print(f"Deleted {deleted}/{len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
