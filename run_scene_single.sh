scene=$1
sentinel_type=$2
# Path to your script or command to run
script_path="ViCo/meeting_challenge/experiment_scripts/run_single.sh"

for job_id in {1..2}; do
  echo "Running job_id=$job_id for scene=$scene"

  salloc -p gpu-preempt -G 1 --mem=100G --nodes=1 -t 120 --job-name=s_$scene --constraint="vram80|a40|l40s" srun bash "$script_path" "$scene" "$sentinel_type" "$job_id"
done

# Optional flags you had commented out:
# --enable_indoor_activities
