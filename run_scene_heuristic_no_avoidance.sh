scene=$1
sentinel_type=$2
# Path to your script or command to run
script_path="ViCo/meeting_challenge/experiment_scripts/run_heuristic_no_avoidance.sh"

for job_id in {1..1}; do
  echo "Running job_id=$job_id for scene=$scene"

  salloc -p gpu-preempt -G 1 --mem=50G -t 210 --job-name=hn_$scene --nodes=1 srun bash "$script_path" "$scene" "$sentinel_type" "$job_id"
done

# Optional flags you had commented out:
# --enable_indoor_activities
