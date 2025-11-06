sentinel_num=$1

rm -r ViCo/meeting_challenge/results_stationary_${sentinel_num}
rm -r ViCo/meeting_challenge/results_patrol_${sentinel_num}

bash run_method.sh heuristic_no_avoidance stationary "$sentinel_num"
bash run_method.sh heuristic stationary "$sentinel_num"
bash run_method.sh single stationary "$sentinel_num"
bash run_method.sh nav stationary "$sentinel_num"
bash run_method.sh heuristic_no_avoidance patrol "$sentinel_num"
bash run_method.sh heuristic patrol "$sentinel_num"
bash run_method.sh single patrol "$sentinel_num"
bash run_method.sh nav patrol "$sentinel_num"