sentinel_num=$1

rm -r ViCo/meeting_challenge/results_stationary_${sentinel_num}
rm -r ViCo/meeting_challenge/results_patrol_${sentinel_num}

bash run_method.sh center_no_avoidance stationary "$sentinel_num"
bash run_method.sh center stationary "$sentinel_num"
bash run_method.sh roco stationary "$sentinel_num"
bash run_method.sh coela stationary "$sentinel_num"
bash run_method.sh sentinel stationary "$sentinel_num"
bash run_method.sh sentinel_no_retry stationary "$sentinel_num"
bash run_method.sh center_no_avoidance patrol "$sentinel_num"
bash run_method.sh center patrol "$sentinel_num"
bash run_method.sh roco patrol "$sentinel_num"
bash run_method.sh coela patrol "$sentinel_num"
bash run_method.sh sentinel patrol "$sentinel_num"
bash run_method.sh sentinel_no_retry patrol "$sentinel_num"