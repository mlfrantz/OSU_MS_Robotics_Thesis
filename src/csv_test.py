#!/home/mlfrantz/miniconda2/bin/python3.6

import csv
import random
import os

filename = '/home/mlfrantz/Documents/MIP_Research/mip_research/data/test.csv'
var_1 = random.random()
var_2 = random.random()+1

check_empty = os.stat(filename).st_size == 0

with open(filename, 'a', newline='') as csvfile:
    fieldnames = ['one', 'two']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if check_empty:
        print("File is empty")
        writer.writeheader()
    writer.writerow({'one':var_1, 'two':var_2})
    writer.writerow({'one':var_1, 'two':var_2})
