agent_num=$1
sentinel_num=$2
gt=$3

rm -r ViCo/meeting_challenge/results_${gt}/${agent_num}_stationary_${sentinel_num}
rm -r ViCo/meeting_challenge/results_${gt}/${agent_num}_patrol_${sentinel_num}

bash run_method.sh center_no_avoidance "$gt" "$agent_num" stationary "$sentinel_num"
bash run_method.sh center "$gt" "$agent_num" stationary "$sentinel_num"
bash run_method.sh roco "$gt" "$agent_num" stationary "$sentinel_num"
bash run_method.sh coela "$gt" "$agent_num" stationary "$sentinel_num"
bash run_method.sh sentinel "$gt" "$agent_num" stationary "$sentinel_num"
bash run_method.sh sentinel_no_refine "$gt" "$agent_num" stationary "$sentinel_num"
bash run_method.sh center_no_avoidance "$gt" "$agent_num" patrol "$sentinel_num"
bash run_method.sh center "$gt" "$agent_num" patrol "$sentinel_num"
bash run_method.sh roco "$gt" "$agent_num" patrol "$sentinel_num"
bash run_method.sh coela "$gt" "$agent_num" patrol "$sentinel_num"
bash run_method.sh sentinel "$gt" "$agent_num" patrol "$sentinel_num"
bash run_method.sh sentinel_no_refine "$gt" "$agent_num" patrol "$sentinel_num"