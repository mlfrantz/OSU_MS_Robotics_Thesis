#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
SIM='cfg/sim_test.yaml'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/team_test_heterogenous.csv'
# outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'

experiment_name='team_test_heterogenous'

SIMXY=('29.00 -91.7' '28.8 -92.7')
# '28.75 -93.25' '25.00 -96.50' '28.82 -91.19'

# These will loop through the 9 starting points for each map

tests=('-n 2 -s 1 1 9 9 -r glider1 rose1' '-n 4 -s 1 1 9 9 -r glider1 rose1'\
 '-n 2 -s 1 1 9 9 5 5 -r glider1 glider2 rose1' '-n 4 -s 1 1 9 9 5 5 -r glider1 glider2 rose1'\
 '-n 2 -s 1 1 9 9 1 9 9 1 -r glider1 glider2 rose1 rose2' '-n 4 -s 1 1 9 9 1 9 9 1 -r glider1 glider2 rose1 rose2'\
 '-n 2 -s 1 1 9 9 1 9 9 1 1 5 9 5 -r glider1 glider2 glider3 glider4 rose1 rose2'\
 '-n 2 -s 1 1 9 9 1 9 9 1 1 5 9 5 5 1 5 9 -r glider1 glider2 glider3 glider4 rose1 rose2 rose3 rose4')

for xy in "${SIMXY[@]}";
do
  echo $xy
  $PATH/src/map_update.py $xy
  for t in "${tests[@]}";
  do
    echo $t
    $PATH/src/mip_test.py $t -o $outfile_path --experiment_name $experiment_name --sim_cfg $SIM -t 3600
    $PATH/src/greedy.py $t -o $outfile_path --experiment_name $experiment_name --sim_cfg $SIM
    $PATH/src/mcts.py $t -o $outfile_path --experiment_name $experiment_name --sim_cfg $SIM
  done
done
