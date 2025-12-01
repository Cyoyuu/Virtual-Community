CONFIG_PATH=assets/scenes/demos/2/
# there is an outdoor demo example: CONFIG_PATH=assets/scenes/demos/1/
OUTPUT_DIR=output_simple/
STEP_LIMIT=5

python simple_env.py --output_dir ${OUTPUT_DIR} \
                          --config_path ${CONFIG_PATH} \
                          --overwrite \
                          --load_indoor_objects \
                          --step_limit ${STEP_LIMIT}

# use --use_luisa_renderer to enable Luisa
