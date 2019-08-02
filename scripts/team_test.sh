#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/team_test_homogenous_map5.csv'
# outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'

experiment_name='team_test_homogenous'

# These will loop through the 9 starting points for each map
x1='1'
y1='1'
x2='9'
y2='9'
x3='5'
y3='5'
x4='1'
y4='9'
x5='9'
y5='1'
x6='1'
y6='5'
x7='9'
y7='5'

budget='2 4'
# budget='10 12'

for b in $budget
do
#   for x in $xPoints
#   do
#     for y in $yPoints
#     do
# echo $b $x $y
  $PATH/src/mip_test.py -n $b -s $x1 $y1 $x2 $y2 $x4 $y4 $x5 $y5 $x6 $y6 $x7 $y7 -r glider1 glider2 glider3 glider4 glider5 glider6 -o $outfile_path --experiment_name $experiment_name
  $PATH/src/greedy.py -n $b -s $x1 $y1 $x2 $y2 $x4 $y4 $x5 $y5 $x6 $y6 $x7 $y7 -r glider1 glider2 glider3 glider4 glider5 glider6 -o $outfile_path --experiment_name $experiment_name
  $PATH/src/mcts.py -n $b -s $x1 $y1 $x2 $y2 $x4 $y4 $x5 $y5 $x6 $y6 $x7 $y7 -r glider1 glider2 glider3 glider4 glider5 glider6 -o $outfile_path --experiment_name $experiment_name
#     done
#   done
done
