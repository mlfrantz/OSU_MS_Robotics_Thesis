#!/home/mlfrantz/miniconda2/bin/python3.6

"""
This code is specifically for plotting the results of my research testing.
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


dataPath = '/home/mlfrantz/Documents/MIP_Research/mip_research/data/'

assert type(sys.argv[1]) is str, "File name must be a string"

df = pd.read_csv(dataPath + sys.argv[1])

nsewConstraint = ['Algorithm', 'Start Point', 'Score', 'Run Time (sec)', 'Budget', 'Budget (hours)', 'Constraints']

df = df.filter(nsewConstraint)
df = df.loc[df['Constraints'] == '_nsew']
df.astype({'Score': 'float64'}).dtypes
print(df.head())
# print(df.loc[df['Constraints'] == '_nsew'])

sns.set(style="ticks", palette="pastel")

# # Load the example tips dataset
# tips = sns.load_dataset("tips")
#
# # Draw a nested boxplot to show bills by day and time
# sns.boxplot(x="day", y="total_bill",
#             hue="smoker", palette=["m", "g"],
#             data=tips)
# sns.despine(offset=10, trim=True)

sns.boxplot(x="Start Point", y="Score",
            hue="Algorithm", palette=["m", "g", "b"],
            data=df)
sns.despine(offset=10, trim=True)

plt.show()
