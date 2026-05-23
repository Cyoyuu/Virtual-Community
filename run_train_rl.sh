#!/bin/bash
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --time=480
#SBATCH --constraint=[a16|l4|a40|l40s|gh200]
#SBATCH --job-name=rl_train
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --output=meeting_challenge/checkpoints/slurm-%j.out

# PPO training driver for the RL meeting-challenge baseline.
#
# Submit:
#   sbatch run_train_rl.sh <agent_num> <sentinel_type> <sentinel_num> <total_episodes> <scenes...>
# Example:
#   sbatch run_train_rl.sh 5 patrol 10 500 AUSTIN NY PARIS
#
# Resumability: train_rl.py auto-detects
#   meeting_challenge/checkpoints/training_state.pt
# and continues from the last completed episode. So if the 480-min slot is
# killed (preemption or wall-time), just `sbatch run_train_rl.sh ...` again
# with the same args -- it picks up where it left off, losing at most one
# in-progress episode.

agent_num=${1:-5}
sentinel_type=${2:-stationary}
sentinel_num=${3:-10}
total_episodes=${4:-10}
shift 4 || true
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
    scenes=("AUSTIN" "DETROIT" "LONDON")
fi

save_dir="meeting_challenge/checkpoints"
mkdir -p "$save_dir"

module load conda/latest
conda activate vico_nav

cd /work/pi_chuangg_umass_edu/xiangye/Virtual-Community
export PYTHONPATH=${PWD}

echo "[run_train_rl] node=$(hostname) job=$SLURM_JOB_ID"
echo "[run_train_rl] scenes=${scenes[*]}"
echo "[run_train_rl] agent_num=$agent_num sentinel=${sentinel_type}/${sentinel_num} total_episodes=$total_episodes"

# Pre-emptive resubmit on imminent wall-time kill.
# SLURM sends SIGUSR1 90s before the wall-time SIGTERM (due to --signal=B:USR1@90).
# Catch it in the bash wrapper (the `B:` prefix means the *batch script* shell
# gets the signal, not the srun step). Submit the next job before letting the
# python step die naturally, so the chain continues without manual sbatch.
SUBMITTED_NEXT=0
on_usr1() {
    if [ "$SUBMITTED_NEXT" -eq 0 ]; then
        SUBMITTED_NEXT=1
        echo "[run_train_rl] caught SIGUSR1; submitting next slot before kill..."
        sbatch "$0" "$agent_num" "$sentinel_type" "$sentinel_num" \
                    "$total_episodes" "${scenes[@]}"
    fi
}
trap on_usr1 USR1

# `-u` forces python unbuffered so per-episode `[train_rl]` prints flush to the
# slurm-*.out file in real time. Genesis logs use its own logger, but the
# trainer's own prints go through stdout.
srun python -u meeting_challenge/train_rl.py \
    --scenes "${scenes[@]}" \
    --agent_num "$agent_num" \
    --sentinel_type "$sentinel_type" \
    --sentinel_num "$sentinel_num" \
    --total_episodes "$total_episodes" \
    --batch_episodes 4 \
    --step_limit 1500 \
    --enable_danger_zone \
    --save_dir "$save_dir" &
SRUN_PID=$!
wait "$SRUN_PID"
status=$?
echo "[run_train_rl] training exited with status=$status"

# If the SIGUSR1 trap already queued the next slot we skip this block. This
# block only fires on clean python exit (training finished or python crashed).
if [ "$SUBMITTED_NEXT" -eq 1 ]; then
    echo "[run_train_rl] next slot already queued by SIGUSR1 handler"
    exit 0
fi
if [ -f "$save_dir/training_state.pt" ]; then
    # Inspect ep_count vs total_episodes -- cheap python eval to decide whether
    # to resubmit. Avoids an infinite resubmit loop once training is done.
    remaining=$(python - <<PY
import torch, sys
try:
    blob = torch.load("$save_dir/training_state.pt", map_location="cpu", weights_only=False)
    print(max(0, $total_episodes - int(blob.get("ep_count", 0))))
except Exception:
    print(0)
PY
)
    echo "[run_train_rl] remaining episodes=$remaining"
    if [ "$remaining" -gt 0 ]; then
        echo "[run_train_rl] resubmitting for next slot..."
        sbatch "$0" "$agent_num" "$sentinel_type" "$sentinel_num" "$total_episodes" "${scenes[@]}"
    else
        echo "[run_train_rl] training complete; not resubmitting."
    fi
fi
