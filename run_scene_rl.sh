scene=$1
gt=$2
agent_num=$3
sentinel_type=$4
sentinel_num=$5
# Optional 6th positional arg: checkpoint path. Defaults to latest.pt.
rl_ckpt=${6:-meeting_challenge/checkpoints/latest.pt}

# Path to your script or command to run
script_path="meeting_challenge/${gt}_scripts/run_rl.sh"

if [ "$sentinel_num" -eq 20 ]; then
  time_limit=180
elif [ "$sentinel_num" -eq 10 ]; then
  time_limit=150
elif [ "$sentinel_num" -eq 5 ]; then
  time_limit=120
else
  time_limit=90
fi
if [ "$gt" = "no_gt" ]; then
  time_limit=$((time_limit + 30))
fi
if [ "$agent_num" -eq 10 ]; then
  time_limit=$((time_limit + 60))
fi
if [ "$time_limit" -gt "480" ]; then
  time_limit=480
fi

# Eval jobs 3..5 keep the same job_id space as the other baselines, so
# cal_results.py picks them up alongside mcts/sentinel/etc.
for job_id in {1..6}; do
  echo "Running job_id=$job_id for scene=$scene (ckpt=$rl_ckpt)"

  salloc -p gpu,gpu-preempt -G 1 --mem=100G --nodes=1 -t "$time_limit" --job-name=rl_$scene --constraint="[a16|a40|gh200|l40s|l4]" srun bash "$script_path" "$scene" "$agent_num" "$sentinel_type" "$sentinel_num" "$job_id" "$rl_ckpt"
done
