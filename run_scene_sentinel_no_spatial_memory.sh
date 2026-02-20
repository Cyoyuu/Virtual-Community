scene=$1
gt=$2
agent_num=$3
sentinel_type=$4
sentinel_num=$5
# Path to your script or command to run
script_path="meeting_challenge/${gt}_scripts/run_sentinel_no_spatial_memory.sh"

if [ "$sentinel_num" -eq 20 ]; then
  time_limit=300
elif [ "$sentinel_num" -eq 10 ]; then
  time_limit=240
elif [ "$sentinel_num" -eq 5 ]; then
  time_limit=180
else
  time_limit=120  # default fallback
fi
if [ "$gt" = "no_gt" ]; then
  time_limit=$((time_limit + 60))
fi
if [ "$time_limit" -gt "480" ]; then
  time_limit=480
fi

for job_id in {1..2}; do
  echo "Running job_id=$job_id for scene=$scene"

  salloc -p gpu,gpu-preempt -G 1 --mem=100G --nodes=1 -t "$time_limit" --job-name=sas_$scene --constraint="[a16|a40|gh200|l40s|l4]" srun bash "$script_path" "$scene" "$agent_num" "$sentinel_type" "$sentinel_num" "$job_id"
done

# Optional flags you had commented out:
# --enable_indoor_activities
