#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
SIM='cfg/sim_test.yaml'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/team_test_sync.csv'
# outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'

experiment_name='team_test_sync'

SIMXY=('29.00 -91.7' '28.8 -92.7' '28.75 -93.25' '25.00 -96.50' '28.82 -91.19')

# These will loop through the 9 starting points for each map

tests=('-n 1 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4'\
 '-n 2 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4'\
 '-n 4 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4'\
 '-n 6 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4'\
 '-n 1 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4 --sync ns'\
 '-n 2 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4 --sync ns'\
 '-n 4 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4 --sync ns'\
 '-n 6 -s 0 0 3 0 7 0 10 0 -t 60 -r glider1 glider2 glider3 glider4 --sync ns')

for xy in "${SIMXY[@]}";
do
  echo $xy
  $PATH/src/map_update.py $xy
  for t in "${tests[@]}";
  do
    echo $t
    $PATH/src/mip_test.py $t -o $outfile_path --experiment_name $experiment_name --sim_cfg $SIM
    # $PATH/src/greedy.py $t -o $outfile_path --experiment_name $experiment_name --sim_cfg $SIM
    # $PATH/src/mcts.py $t -o $outfile_path --experiment_name $experiment_name --sim_cfg $SIM
  done
done
