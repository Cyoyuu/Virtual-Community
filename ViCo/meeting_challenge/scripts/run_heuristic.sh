export PYTHONPATH=${PWD}

# export keep_running='1'

python ViCo/meeting_challenge/challenge.py --head_less \
--backend gpu \
--multi_process \
--skip_avatar_animation \
--enable_gt_segmentation \
--output_dir ViCo/meeting_challenge/output \
--scene DETROIT \
--enable_outdoor_objects \
--outdoor_objects_max_num 5 \
--resolution 512 \
--config agents_num_15_with_schedules \
--agent_type heuristic \
--save_per_seconds 50 \
--step_limit 1500 \
--lm_source azure \
--lm_id gpt-4o \
--debug \
--overwrite

# --enable_indoor_scene \
# --enable_indoor_activities \