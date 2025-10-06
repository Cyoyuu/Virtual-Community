scene=$1
job_id=$2

export PYTHONPATH=${PWD}

python ViCo/meeting_challenge/challenge.py --head_less \
--backend gpu \
--multi_process \
--skip_avatar_animation \
--enable_gt_segmentation \
--output_dir ViCo/meeting_challenge/output \
--scene "${scene}" \
--job_id "${job_id}" \
--enable_outdoor_objects \
--enable_indoor_scene \
--outdoor_objects_max_num 5 \
--resolution 512 \
--config agents_num_5 \
--agent_type heuristic_nav \
--save_per_seconds 200 \
--step_limit 2000 \
--lm_source azure \
--lm_id gpt-4o \
--debug \
--overwrite

# Optional flags you had commented out:
# --enable_indoor_activities
