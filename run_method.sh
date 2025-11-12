#!/bin/bash
agent_type=$1
sentinel_type=$2
sentinel_num=$3

# Define arrays of scenes and task types
# scenes=("MADRID" "HARVARD" "AMSTERDAM" "AUSTIN" "BALTIMORE" "BELGRADE" "BERLIN" "BRATISLAVA" "BRUSSELS" "BUDAPEST" "CALGARY" "CHRISTCHURCH" "COLUMBUS" "DENVER" "DETROIT" "EL_PASO" "FLORENCE" "FORT_WORTH" "FRANKFURT" "HAMBURG" "LONGISLAND" "MADISON" "BARCELONA" "LONDON")
scenes=("MADRID" "HARVARD" "AMSTERDAM" "AUSTIN" "BERLIN" "CALGARY" "COLUMBUS" "DENVER" "DETROIT" "EL_PASO" "HAMBURG" "LONGISLAND" "MADISON" "BARCELONA" "LONDON")
#scenes=("AMSTERDAM" "AUSTIN" "BARCELONA" "BERLIN" "BRATISLAVA" "BRUSSELS" "BUDAPEST" "CALGARY" "CHRISTCHURCH" "DENVER" "DETROIT")
#scenes=("$1")
echo ${scenes[@]}
#task_types=("transport" "deliver" "search")
task_types=("collect")
#task_types=("$2")
echo ${task_types[@]}

# Path to your script or command to run
script_path="run_scene_$agent_type.sh"

# Iterate over scenes and task types
for scene in "${scenes[@]}"; do
  for task in "${task_types[@]}"; do
    echo "Running for scene: $scene"
    
    # Example: run python script with arguments
    bash "$script_path" "$scene" "$sentinel_type" "$sentinel_num" &
    
    # Optionally check exit status and handle errors
    if [ $? -ne 0 ]; then
      echo "Error running for scene=$scene"
      # Optionally exit or continue
      # exit 1
    fi
  done
done

echo "All tasks completed."

