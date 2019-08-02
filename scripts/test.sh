#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/team_test_heterogenous_map5.csv'
# outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'

experiment_name='team_test_heterogenous'

# These will loop through the 9 starting points for each map

tests=('1 1 9 9 -r glider1 rose1'\
 '1 1 9 9 5 5 -r glider1 glider2 rose1' '1 1 9 9 1 9 9 1 -r glider1 glider2 rose1 rose2'\
 '1 1 9 9 1 9 9 1 1 5 9 5 -r glider1 glider2 glider3 glider4 rose1 rose2'\
 '1 1 9 9 1 9 9 1 1 5 9 5 5 1 5 9 -r glider1 glider2 glider3 glider4 rose1 rose2 rose3 rose4')
test=('1 1 9 9 1 9 9 1 -r glider1 glider2 rose1 rose2')

budget='4'
# budget='10 12'

for t in "${test[@]}";
do
  for b in $budget
  do
  #   for x in $xPoints
  #   do
  #     for y in $yPoints
  #     do
  # echo $b $x $y
    # echo $t
    # $PATH/src/mip_test.py -n $b -s $t -o $outfile_path --experiment_name $experiment_name -t 3600
    $PATH/src/greedy.py -n $b -s $t -o $outfile_path --experiment_name $experiment_name
    $PATH/src/mcts.py -n $b -s $t -o $outfile_path --experiment_name $experiment_name
  #     done
  #   done
  done
done
