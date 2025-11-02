scene=$1
# Path to your script or command to run
script_path="ViCo/meeting_challenge/experiment_scripts/run_heuristic.sh"

for job_id in {1..1}; do
  echo "Running job_id=$job_id for scene=$scene"

  salloc -p gpu-preempt -G 1 --mem=50G -t 180 --job-name=h_$scene --nodes=1 srun bash "$script_path" "$scene" "$job_id"
done

# Optional flags you had commented out:
# --enable_indoor_activities
