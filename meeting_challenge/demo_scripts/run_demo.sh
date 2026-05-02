scene=$1
agent_type=$2
agent_num=$3
sentinel_type=$4
sentinel_num=$5
job_id=$6

export PYTHONPATH=${PWD}

# in replay mode, --output_dir serves as the source dir
python meeting_challenge/challenge.py --head_less \
--backend gpu \
--multi_process \
--skip_avatar_animation \
--enable_gt_segmentation \
--output_dir meeting_challenge/output \
--scene "${scene}" \
--job_id "${job_id}" \
--enable_outdoor_objects \
--enable_indoor_scene \
--outdoor_objects_max_num 5 \
--resolution 512 \
--config agents_num_2_demo \
--agent_type ${agent_type} \
--agent_num ${agent_num} \
--sentinel_type ${sentinel_type} \
--sentinel_num ${sentinel_num} \
--enable_danger_zone \
--save_per_seconds 1 \
--step_limit 1500 \
--use_luisa_renderer \
--lm_source azure \
--lm_id gpt-4o \
--debug \
--overwrite \
--enable_demo_camera

# Optional flags you had commented out:
# --enable_indoor_activities