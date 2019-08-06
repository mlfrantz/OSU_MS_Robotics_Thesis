#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/constraint_test_map4.csv'
# outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'

experiment_name='Constraint_Test'

# These will loop through the 9 starting points for each map
xPoints='1 5 9'
yPoints='1 5 9'

budget='1 2 4 8'
# budget='10 12'

constraint=('-d nsew' '-d diag' '--anti_curl' '--force_curl' '--same_point')

for c in "${constraint[@]}"
do
  for b in $budget
  do
    for x in $xPoints
    do
      for y in $yPoints
      do
        echo $c $b $x $y
        $PATH/src/mip_test.py -n $b -s $x $y -t 1800 -r glider1 -o $outfile_path --experiment_name $experiment_name $c
        # $PATH/src/greedy.py -n $b -s $x $y -r glider1 -o $outfile_path --experiment_name $experiment_name
        # $PATH/src/mcts.py -n $b -s $x $y -r glider1 -o $outfile_path --experiment_name $experiment_name
      done
    done
  done
done
