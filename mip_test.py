#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

# Path lenth Np
Np = int(sys.argv[1])
time = range(Np)
start = (8, 8)

# Problem data, matrix transposed to allow for proper x,y coordinates to be mapped wih i,j
field = np.genfromtxt('test_field_2.csv', delimiter=',', dtype=float).transpose()
DX = np.arange(field.shape[0]) # Integer values for range of X coordinates
DY = np.arange(field.shape[1]) # Integer values for range of Y coordinates
numX = len(DX) # Number of X Positions
numY = len(DY) # Number of Y Positions

m = Model()
m.Params.TIME_LIMIT = 60.0
# m.Params.MIPGap = 0.01

x = m.addVars(time, lb=DX[0], ub=DX[-1], vtype=GRB.INTEGER, name='x')
y = m.addVars(time, lb=DY[0], ub=DY[-1], vtype=GRB.INTEGER, name='y')
f = m.addVars(time, lb=np.min(field), ub=np.max(field), vtype=GRB.CONTINUOUS, name='f')

m.addConstr((x[time[0]] == start[0]), name="Initial x")
m.addConstr((y[time[0]] == start[1]), name="Initial y")
m.addConstr((f[time[0]] == field[start[0], start[1]]), name="Initial f")


pos = m.addVars(time, DX, DY, vtype=GRB.BINARY, name='pos')

# m.addConstrs((pos.sum(i,j,'*') == 1 for i in range(numX) for j in range(numY)))
# # m.addConstrs(pos.sum(i,j) == 1 for i in range(numX) for j in range(numY))
# # m.addConstrs(pos.sum('*',j) == 1 for j in range(numY))
# #m.addConstrs( (pos[i, j] == 0 for i in range(numX) for j in range(numY) ), name="Initial zeros")
# pos[start[0], start[1], 0] = 1

# for t in time:
#     for i in range(numX):
#         for j in range(numY):
#             if i == x[t] and j == y[t]:
#                 pos[t,i,j] == 1

for t in time[1:]:
    m.addConstr(quicksum(pos[t,i,j] for i in range(numX) for j in range(numY)) == t+1)

# m.addConstrs(pos[i,j] == 1 for i in range(numX) for j in range(numY))
#
# m.update()
# print(pos)

lx = m.addVars(time, DX, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='lx')
ly = m.addVars(time, DY, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='ly')
lxy = m.addVars(time, DX, DY, vtype=GRB.CONTINUOUS, name='lxy')

for t in time:
    m.addSOS(GRB.SOS_TYPE2, [lx[t,i] for i in range(numX)])
    m.addSOS(GRB.SOS_TYPE2, [ly[t,j] for j in range(numY)])

for t in time:
    m.addConstrs(quicksum(lxy[t, i, j] for j in range(numY)) == lx[t, i] for i in range(numX))
    m.addConstrs(quicksum(lxy[t, i, j] for i in range(numX)) == ly[t, j] for j in range(numY))

m.addConstrs((quicksum(lx[t, i] for i in range(numX)) == 1 for t in time))
m.addConstrs((quicksum(ly[t, j] for j in range(numY)) == 1 for t in time))

m.addConstrs((quicksum(DX[i]*lx[t,i] for i in range(numX)) == x[t] for t in time))
m.addConstrs((quicksum(DY[j]*ly[t,j] for j in range(numY)) == y[t] for t in time))
m.addConstrs((quicksum(field[i,j]*lxy[t,i,j] for i in range(numX) for j in range(numY)) == f[t] for t in time))

# Binary variables for motion constraints
b_range = range(4)
b = m.addVars(time, b_range, vtype=GRB.BINARY, name='b')

# Primary Motion constraints
m.addConstrs(x[t-1] + b[t,0] - b[t,1] == x[t] for t in time[1:])
m.addConstrs(b[t,0] + b[t,1] <= 1 for t in time[1:])
m.addConstrs(y[t-1] + b[t,2] - b[t,3] == y[t] for t in time[1:])
m.addConstrs(b[t,2] + b[t,3] <= 1 for t in time[1:])
m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] >= 1 for t in time[1:])

# PCA/Eignevector/Cardinal constraints
# m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] == 1 for t in time[1:]) #PCA N-S-E-W
#m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] == 2 for t in time if t!=0) #PCA Diag

# Curling constraints go from [2,...,Np] and enforce curing
# delta = 1
# t_delta = 2
# m.addConstrs(x[t-t_delta] - x[t] <= delta for t in time[t_delta:])
# m.addConstrs(x[t] - x[t-t_delta] <= delta for t in time[t_delta:])
# m.addConstrs(y[t-t_delta] - y[t] <= delta for t in time[t_delta:])
# m.addConstrs(y[t] - y[t-t_delta] <= delta for t in time[t_delta:])

# Straight path constraints
# Not working
# path_len = 3
# pt_delta = 3
# m.addConstrs(x[t-pt_delta] - x[t] <= path_len for t in time[pt_delta:])
# m.addConstrs(x[t] - x[t-pt_delta] <= path_len for t in time[pt_delta:])
# m.addConstrs(y[t-pt_delta] - y[t] <= path_len for t in time[pt_delta:])
# m.addConstrs(y[t] - y[t-pt_delta] <= path_len for t in time[pt_delta:])

m.update()

obj = quicksum(f[t] for t in time)
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()

# path = np.zeros(Np)
# print(m.getAttr('X', pos).values())
# for v in m.getVars():
#     if v.X == 0:
#         print("%s %f" % (v.varName, v.X))

path_x = m.getAttr('X', x).values()
path_y = m.getAttr('X', y).values()
print('Obj: %g' % obj.getValue())
path = list(zip(path_x, path_y))
print(path)
#path = list(zip(path_x.values(),path_y.values()))
#print(path)
plt.imshow(field.transpose(), interpolation='gaussian', cmap= 'gnuplot')
plt.colorbar()
plt.plot(path_x, path_y, color='c', linewidth=2.0)
plt.plot(path_x[0], path_y[0], color='g', marker='o')
plt.plot(path_x[-1], path_y[-1], color='r', marker='o')

points = []
for i, point in enumerate(path):
    points.append(points)
    if point in path and point not in point:
        plt.annotate(i+1, (path[i][0], path[i][1]))


plt.show()
