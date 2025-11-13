agent_num=$1
sentinel_num=$2

rm -r ViCo/meeting_challenge/results_${agent_num}_stationary_${sentinel_num}
rm -r ViCo/meeting_challenge/results_${agent_num}_patrol_${sentinel_num}

bash run_method.sh center_no_avoidance "$agent_num" stationary "$sentinel_num"
bash run_method.sh center "$agent_num" stationary "$sentinel_num"
bash run_method.sh roco "$agent_num" stationary "$sentinel_num"
bash run_method.sh coela "$agent_num" stationary "$sentinel_num"
bash run_method.sh sentinel "$agent_num" stationary "$sentinel_num"
bash run_method.sh sentinel_no_retry "$agent_num" stationary "$sentinel_num"
bash run_method.sh center_no_avoidance "$agent_num" patrol "$sentinel_num"
bash run_method.sh center "$agent_num" patrol "$sentinel_num"
bash run_method.sh roco "$agent_num" patrol "$sentinel_num"
bash run_method.sh coela "$agent_num" patrol "$sentinel_num"
bash run_method.sh sentinel "$agent_num" patrol "$sentinel_num"
bash run_method.sh sentinel_no_retry "$agent_num" patrol "$sentinel_num"