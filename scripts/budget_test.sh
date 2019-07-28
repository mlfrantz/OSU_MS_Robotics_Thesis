#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/budget_test_map5.csv'
# outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'

experiment_name='Budget_Test'

# These will loop through the 9 starting points for each map
xPoints='1 5 9'
yPoints='1 5 9'

budget='1 2 4 8'
# budget='10 12'

for b in $budget
do
  for x in $xPoints
  do
    for y in $yPoints
    do
      echo $b $x $y
      $PATH/src/mip_test.py -n $b -s $x $y -t 3600 -r glider1 -o $outfile_path --experiment_name $experiment_name
      $PATH/src/greedy.py -n $b -s $x $y -r glider1 -o $outfile_path --experiment_name $experiment_name
      $PATH/src/mcts.py -n $b -s $x $y -r glider1 -o $outfile_path --experiment_name $experiment_name
    done
  done
done
