#!/bin/bash
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --constraint=[a16|l4|a40|l40s|gh200]
#SBATCH --job-name=mat_train
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --output=meeting_challenge/checkpoints_mat/slurm-%j.out

# PPO training driver for the MAT meeting-challenge baseline.
#
# Submit:
#   sbatch run_train_mat.sh <agent_num> <sentinel_type> <sentinel_num> <total_episodes> <scenes...>
# Example:
#   sbatch run_train_mat.sh 5 stationary 10 200 AUSTIN DETROIT LONDON
#
# Auto-resumable: train_mat.py picks up
#   meeting_challenge/checkpoints_mat/training_state.pt
# on every restart, losing at most one in-progress episode.

agent_num=${1:-5}
sentinel_type=${2:-stationary}
sentinel_num=${3:-10}
total_episodes=${4:-20}
shift 4 || true
scenes=("$@")
if [ ${#scenes[@]} -eq 0 ]; then
    scenes=("AUSTIN" "DETROIT" "LONDON")
fi

save_dir="meeting_challenge/checkpoints_mat"
mkdir -p "$save_dir"

module load conda/latest
conda activate vico_nav

cd /work/pi_chuangg_umass_edu/xiangye/Virtual-Community
export PYTHONPATH=${PWD}
export PYTHONUNBUFFERED=1

echo "[run_train_mat] node=$(hostname) job=$SLURM_JOB_ID"
echo "[run_train_mat] scenes=${scenes[*]}"
echo "[run_train_mat] agent_num=$agent_num sentinel=${sentinel_type}/${sentinel_num} total_episodes=$total_episodes"

SUBMITTED_NEXT=0
on_usr1() {
    if [ "$SUBMITTED_NEXT" -eq 0 ]; then
        SUBMITTED_NEXT=1
        echo "[run_train_mat] caught SIGUSR1; submitting next slot before kill..."
        sbatch "$0" "$agent_num" "$sentinel_type" "$sentinel_num" \
                    "$total_episodes" "${scenes[@]}"
    fi
}
trap on_usr1 USR1

srun python -u meeting_challenge/train_mat.py \
    --scenes "${scenes[@]}" \
    --agent_num "$agent_num" \
    --sentinel_type "$sentinel_type" \
    --sentinel_num "$sentinel_num" \
    --total_episodes "$total_episodes" \
    --batch_episodes 4 \
    --step_limit 1500 \
    --planning_interval 50 \
    --enable_danger_zone \
    --save_dir "$save_dir" &
SRUN_PID=$!
wait "$SRUN_PID"
status=$?
echo "[run_train_mat] training exited with status=$status"

if [ "$SUBMITTED_NEXT" -eq 1 ]; then
    echo "[run_train_mat] next slot already queued by SIGUSR1 handler"
    exit 0
fi
if [ -f "$save_dir/training_state.pt" ]; then
    remaining=$(python - <<PY
import torch
try:
    blob = torch.load("$save_dir/training_state.pt", map_location="cpu", weights_only=False)
    print(max(0, $total_episodes - int(blob.get("ep_count", 0))))
except Exception:
    print(0)
PY
)
    echo "[run_train_mat] remaining episodes=$remaining"
    if [ "$remaining" -gt 0 ]; then
        echo "[run_train_mat] resubmitting for next slot..."
        sbatch "$0" "$agent_num" "$sentinel_type" "$sentinel_num" "$total_episodes" "${scenes[@]}"
    else
        echo "[run_train_mat] training complete; not resubmitting."
    fi
fi
