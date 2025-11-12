scene=$1
sentinel_type=$2
sentinel_num=$3
# Path to your script or command to run
script_path="ViCo/meeting_challenge/experiment_scripts/run_sentinel_no_refine.sh"

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

  salloc -p gpu-preempt -G 1 --mem=100G --nodes=1 -t "$time_limit" --job-name=s_$scene srun bash "$script_path" "$scene" "$sentinel_type" "$sentinel_num" "$job_id"
done

# Optional flags you had commented out:
# --enable_indoor_activities
