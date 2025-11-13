scene=$1
agent_num=$2
sentinel_type=$3
sentinel_num=$4
# Path to your script or command to run
script_path="ViCo/meeting_challenge/experiment_scripts/run_center.sh"

if [ "$sentinel_num" -eq 20 ]; then
  time_limit=270
elif [ "$sentinel_num" -eq 10 ]; then
  time_limit=210
elif [ "$sentinel_num" -eq 5 ]; then
  time_limit=150
else
  time_limit=120  # default fallback
fi

for job_id in {1..1}; do
  echo "Running job_id=$job_id for scene=$scene"

  salloc -p gpu-preempt -G 1 --mem=50G -t "$time_limit" --job-name=h_$scene --nodes=1 srun bash "$script_path" "$scene" "$agent_num" "$sentinel_type" "$sentinel_num" "$job_id"
done

# Optional flags you had commented out:
# --enable_indoor_activities
