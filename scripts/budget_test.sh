#!/bin/bash

PATH='/home/mlfrantz/Documents/MIP_Research/mip_research'
outfile_path='/home/mlfrantz/Documents/MIP_Research/mip_research/data/budget_test.csv'
experiment_name='Budget_Test'

# These will lopp through the 9 starting points for each map
xPoints='1 5 9'
yPoints='1 5 9'

budget='8 10 12'
#'1 2 4'

for b in $budget
do
  for x in $xPoints
  do
    for y in $yPoints
    do
      echo $b $x $y
      $PATH/src/mip_test.py -n $b -s $x $y -t 3600 -r glider1 -o $outfile_path --experiment_name $experiment_name
    done
  done
done
