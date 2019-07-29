#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/team_test_homogenous_map5.csv'
# outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'

experiment_name='Budget_Test'

# These will loop through the 9 starting points for each map
x1='1'
y1='1'
x2='9'
y2='9'

budget='2'
# budget='10 12'

# for b in $budget
# do
#   for x in $xPoints
#   do
#     for y in $yPoints
#     do
# echo $b $x $y
$PATH/src/mip_test.py -n $budget -s $x1 $y1 $x2 $y2 -t 3600 -r glider1 glider2 -o $outfile_path --experiment_name $experiment_name
$PATH/src/greedy.py -n $budget -s $x1 $y1 $x2 $y2  -r glider1 glider2 -o $outfile_path --experiment_name $experiment_name
$PATH/src/mcts.py -n $budget -s $x1 $y1 $x2 $y2 -r glider1 glider2 -o $outfile_path --experiment_name $experiment_name
#     done
#   done
# done
