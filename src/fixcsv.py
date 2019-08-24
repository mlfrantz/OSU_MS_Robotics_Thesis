#!/home/mlfrantz/miniconda2/bin/python3.6

"""
I messed up when running constraint testing and accidenttilly added a string to the score.
This script just fixes that mistake.
"""
import sys
import pandas as pd
import pdb

dataPath = '/home/mlfrantz/Documents/MIP_Research/mip_research/data/'

assert type(sys.argv[1]) is str, "File name must be a string"

df = pd.read_csv(dataPath + sys.argv[1])

for index, row in df.iterrows():
    # pdb.set_trace()
    try:
        if row["Constraints"][0] == '_':
            df["Constraints"][index] = row["Constraints"][1:]
            # continue
            # print(index, row['Score'][7:])
            # print(index, df['Score'][index][7:])
    except TypeError:
        df.at['Constraints',index] = 'none'
    # if row['Score'] == '_no_solution':
    #     df['Score'][index] = 0

print(df['Constraints'].tail())

df.to_csv(dataPath + sys.argv[1], index=False)
