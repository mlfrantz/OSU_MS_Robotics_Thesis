#!/home/mlfrantz/miniconda2/bin/python3.6

"""
This code is specifically for plotting the results of my research testing.
"""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
from scipy import stats

def normalized(x, **kws):
    # This function scales the data between 0-1. the 'index' variable is to select
    # a specific time frame to normalize to.
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)

dataPath = '/home/mlfrantz/Documents/MIP_Research/mip_research/data/'

assert type(sys.argv[1]) is str, "File name must be a string"

df = pd.read_csv(dataPath + sys.argv[1])

nsewConstraint = ['Algorithm', 'Start', 'Score', 'Run_Time_(sec)', 'Budget_hours', 'Constraints']

df = df.filter(nsewConstraint)
# df = df.loc[df['Constraints'] == '_nsew']
df.astype({'Score': 'float64'}).dtypes
#
# mip_none = df.query("Algorithm == 'MIP' and Constraints == 'none'")
# mip_same = df.query("Algorithm == 'MIP' and Constraints == 'same_point'")
# mip_nsew = df.query("Algorithm == 'MIP' and Constraints == '_nsew'")
# mip_diag = df.query("Algorithm == 'MIP' and Constraints == '_diag'")
# mip_anti = df.query("Algorithm == 'MIP' and Constraints == '_antiCurl'")
# mip_force = df.query("Algorithm == 'MIP' and Constraints == '_forceCurl'")
# # greedy_mean = df.query("Algorithm == 'Greedy' and Constraints == 'same_point'")
# # mcts_mean = df.query("Algorithm == 'MCTS' and Constraints == 'same_point'")
# # mip_df = pd.concat(mip_none,mip_constraint)
#
# values = []
# budget = [1, 2, 4, 8]
# for b in budget:
#     for a in [mip_none, mip_same, mip_nsew, mip_diag, mip_anti, mip_force]:
#         mean = np.mean(a.loc[a["Budget_hours"] == b])[0]
#         error = np.std(a.loc[a["Budget_hours"] == b])[0]/np.sqrt(len(a.loc[a["Budget_hours"] == b].columns))
#         # print(mean, error)
#         if a is mip_none:
#             algorithm = "None"
#         elif a is mip_same:
#             algorithm = "same_point"
#         elif a is mip_nsew:
#             algorithm = "nsew"
#         elif a is mip_diag:
#             algorithm = "diag"
#         elif a is mip_anti:
#             algorithm = "anti"
#         elif a is mip_force:
#             algorithm = "force"
#
#
#         values.append((algorithm, mean, error, b))
#
# columns = ["Algorithm", "Mean", "Std_Error", "Budget_hours"]
# budget_df = pd.DataFrame(values, columns=columns)
#
# for b in budget:
#     temp = budget_df.loc[budget_df["Budget_hours"] == b]
#     mean_min = np.min(temp["Mean"])
#     std_min = np.min(temp["Std_Error"])
#     temp["Mean"] = (temp["Mean"]/mean_min)*100 - 100
#     budget_df.loc[budget_df["Budget_hours"] == b] = temp
#
# # sns.set(style="ticks", palette="pastel")
# # g = sns.catplot(x="Budget_hours", y="Mean", hue="Algorithm", data=budget_df,
# #                 height=6, kind="bar", palette="bright")
# # g.despine(left=True)
# # g.set_ylabels("Average Percent Improved")
# # # plt.show()
#
# width = 1/7
# fig, ax = plt.subplots()
# plt.rcParams.update({'font.size':14})
# # plt.rc('font', size=20)
#
# budget = np.array(budget)
# # print(budget_df)
# # rect_mip = ax.bar(budget-width/2, budget_df.loc[budget_df["Algorithm"] == "None"]["Mean"], width, color='b', yerr=budget_df.loc[budget_df["Algorithm"] == "None"]["Std_Error"])
# # rect_constraint = ax.bar(budget+width/2, budget_df.loc[budget_df["Algorithm"] == "Constraint"]["Mean"], width, color='g', yerr=budget_df.loc[budget_df["Algorithm"] == "Constraint"]["Std_Error"])
# # rect_greedy = ax.bar(budget, budget_df.loc[budget_df["Algorithm"] == "Greedy"]["Mean"], width, color='orange', yerr=budget_df.loc[budget_df["Algorithm"] == "Greedy"]["Std_Error"])
# # rect_mcts = ax.bar(budget+width, budget_df.loc[budget_df["Algorithm"] == "MCTS"]["Mean"], width, color='g', yerr=budget_df.loc[budget_df["Algorithm"] == "MCTS"]["Std_Error"])
# rect_none = ax.bar(budget-2.5*width, budget_df.loc[budget_df["Algorithm"] == "None"]["Mean"], width, color='b', yerr=budget_df.loc[budget_df["Algorithm"] == "None"]["Std_Error"])
# rect_same = ax.bar(budget-1.5*width, budget_df.loc[budget_df["Algorithm"] == "same_point"]["Mean"], width, color='g', yerr=budget_df.loc[budget_df["Algorithm"] == "same_point"]["Std_Error"])
# rect_nsew = ax.bar(budget-width/2, budget_df.loc[budget_df["Algorithm"] == "nsew"]["Mean"], width, color='r', yerr=budget_df.loc[budget_df["Algorithm"] == "same_point"]["Std_Error"])
# rect_diag = ax.bar(budget+width/2, budget_df.loc[budget_df["Algorithm"] == "diag"]["Mean"], width, color='purple', yerr=budget_df.loc[budget_df["Algorithm"] == "same_point"]["Std_Error"])
# rect_anti = ax.bar(budget+1.5*width, budget_df.loc[budget_df["Algorithm"] == "anti"]["Mean"], width, color='orange', yerr=budget_df.loc[budget_df["Algorithm"] == "same_point"]["Std_Error"])
# rect_force = ax.bar(budget+2.5*width, budget_df.loc[budget_df["Algorithm"] == "force"]["Mean"], width, color='y', yerr=budget_df.loc[budget_df["Algorithm"] == "same_point"]["Std_Error"])
#
# ax.set_ylabel("Percent Improved")
# ax.set_xlabel("Budget (Hours)")
# ax.set_xticks(budget)
# # ax.legend((rect_mip[0],rect_greedy[0],rect_mcts[0]), ('MIP', 'Greedy', 'MCTS'))
# ax.legend((rect_none[0],rect_same[0], rect_nsew[0],rect_diag[0],rect_anti[0], rect_force[0]), ('None', 'Same Point Allowed', 'NSEW', "Diagonal Only", 'Anti-Curling', 'Forced Curling'))
#
# plt.show()



# This is a cat and wiskers plot
sns.set(style="ticks", palette="pastel")
sns.boxplot(x="Budget_hours", y="Score",
            hue="Algorithm", data=df.query("Budget_hours < 12"))
sns.despine(offset=10, trim=True)

plt.show()



# This is a box plot
# g = sns.catplot(x="Budget_hours", y="Score", hue="Algorithm", data=df.query("Budget_hours < 12"),
#                 height=6, kind="bar", palette="muted", estimator=np.mean)
# g.despine(left=True)
# g.set_ylabels("survival probability")
#
# plt.show()

# # This is line plot for time
# g = sns.catplot(x="Budget_hours", y="Run_Time_(sec)", hue="Algorithm",
#                 capsize=.2, palette=["orange", "g", "b"], height=6, aspect=.75,
#                 kind="point", data=df.query("Budget_hours < 12"), log=True)
# g.set(yscale="log")
# g.despine(left=True)
# g.set_ylabels("Run Time (Seconds)")
# plt.legend(loc='upper left')
# plt.show()
