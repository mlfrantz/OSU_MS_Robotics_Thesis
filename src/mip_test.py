#!/usr/bin/python

import sys, pdb, time
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *


# Path lenth Np
Np = int(sys.argv[1])
steps = range(Np)
start = (10, 17)
# end = (10, 10)

# Problem data, matrix transposed to allow for proper x,y coordinates to be mapped wih i,j
field = np.genfromtxt('test_fields/test_field_2.csv', delimiter=',', dtype=float).transpose()
DX = np.arange(field.shape[0]) # Integer values for range of X coordinates
DY = np.arange(field.shape[1]) # Integer values for range of Y coordinates

numX = len(DX) # Number of X Positions
numY = len(DY) # Number of Y Positions

m = Model()
m.Params.TIME_LIMIT = 60.0
# m.Params.MIPGap = 0.01

x = m.addVars(steps, lb=DX[0], ub=DX[-1], vtype=GRB.INTEGER, name='x')
y = m.addVars(steps, lb=DY[0], ub=DY[-1], vtype=GRB.INTEGER, name='y')
f = m.addVars(steps, lb=np.min(field), ub=np.max(field), vtype=GRB.CONTINUOUS, name='f')

# Starting position contraint
m.addConstr((x[steps[0]] == start[0]), name="Initial x")
m.addConstr((y[steps[0]] == start[1]), name="Initial y")
m.addConstr((f[steps[0]] == field[start[0], start[1]]), name="Initial f")


# Optional end position constraint.
# m.addConstr((x[steps[-1]] == end[0]), name="End x")
# m.addConstr((y[steps[-1]] == end[1]), name="End y")
# m.addConstr((f[steps[-1]] == field[end[0], end[1]]), name="End f")

# m.update()

lx = m.addVars(steps, DX, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='lx')
ly = m.addVars(steps, DY, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='ly')
lxy = m.addVars(steps, DX, DY, vtype=GRB.CONTINUOUS, name='lxy')

for t in steps:
    m.addSOS(GRB.SOS_TYPE2, [lx[t,i] for i in range(numX)])
    m.addSOS(GRB.SOS_TYPE2, [ly[t,j] for j in range(numY)])

for t in steps:
    m.addConstrs(quicksum(lxy[t, i, j] for j in range(numY)) == lx[t, i] for i in range(numX))
    m.addConstrs(quicksum(lxy[t, i, j] for i in range(numX)) == ly[t, j] for j in range(numY))

m.addConstrs((quicksum(lx[t, i] for i in range(numX)) == 1 for t in steps))
m.addConstrs((quicksum(ly[t, j] for j in range(numY)) == 1 for t in steps))

m.addConstrs((quicksum(DX[i]*lx[t,i] for i in range(numX)) == x[t] for t in steps))
m.addConstrs((quicksum(DY[j]*ly[t,j] for j in range(numY)) == y[t] for t in steps))
m.addConstrs((quicksum(field[i,j]*lxy[t,i,j] for i in range(numX) for j in range(numY)) == f[t] for t in steps))

# Binary variables for motion constraints
b_range = range(4)
b = m.addVars(steps, b_range, vtype=GRB.BINARY, name='b')

# Primary Motion constraints
m.addConstrs(x[t-1] + b[t,0] - b[t,1] == x[t] for t in steps[1:])
m.addConstrs(b[t,0] + b[t,1] <= 1 for t in steps[1:])
m.addConstrs(y[t-1] + b[t,2] - b[t,3] == y[t] for t in steps[1:])
m.addConstrs(b[t,2] + b[t,3] <= 1 for t in steps[1:])
m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] >= 1 for t in steps[1:])

# PCA/Eignevector/Cardinal constraints
# m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] == 1 for t in steps[1:]) #PCA N-S-E-W
# m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] == 2 for t in steps if t!=0) #PCA Diag

# Constraint to prevent gonig through the same point twice
# M = 100
# for t in steps[1:]:
#     t1 = m.addVars(t, range(4), vtype=GRB.BINARY, name='t%d'% t)
#     for s in range(t):
#         m.addConstr(x[t]-x[s] >= 0.1 - M*t1[s,0])
#         m.addConstr(x[s]-x[t] >= 0.1 - M*t1[s,1])
#         m.addConstr(y[t]-y[s] >= 0.1 - M*t1[s,2])
#         m.addConstr(y[s]-y[t] >= 0.1 - M*t1[s,3])
#         m.addConstr(t1[s,0] + t1[s,1] + t1[s,2] + t1[s,3] <= 3)

# Anti-Curling constraints go from [3,...,Np] and enforce curing (this oin works!)
# t_range = range(4)
# t1 = m.addVars(steps, t_range, vtype=GRB.BINARY, name='t1')
# delta = 2
# t_delta = 3
# M = 100
# m.addConstrs(x[t-t_delta] - x[t] >= delta - M*t1[t,0] for t in steps[t_delta:])
# m.addConstrs(x[t] - x[t-t_delta] >= delta - M*t1[t,1] for t in steps[t_delta:])
# m.addConstrs(y[t-t_delta] - y[t] >= delta - M*t1[t,2] for t in steps[t_delta:])
# m.addConstrs(y[t] - y[t-t_delta] >= delta - M*t1[t,3] for t in steps[t_delta:])
# m.addConstrs(t1[t,0] + t1[t,1] + t1[t,2] + t1[t,3] <= 3 for t in steps[t_delta:])

# Curling constraints go from [2,...,Np] and enforce curing
# delta = 2
# t_delta = 3
# m.addConstrs(x[t-t_delta] - x[t] <= delta for t in steps[t_delta:])
# m.addConstrs(x[t] - x[t-t_delta] <= delta for t in steps[t_delta:])
# m.addConstrs(y[t-t_delta] - y[t] <= delta for t in steps[t_delta:])
# m.addConstrs(y[t] - y[t-t_delta] <= delta for t in steps[t_delta:])

# Straight path constraints
# Working better
# path_len_x = 3
# path_len_y = 2
# pt_delta = 4
# M = 100
# t_range = range(4)
# ts = m.addVars(steps, t_range, vtype=GRB.BINARY, name='t1')
# m.addConstrs(x[t-pt_delta] - x[t] >= path_len_x - M*ts[t,0] for t in steps[4:])
# m.addConstrs(x[t] - x[t-pt_delta] >= path_len_x - M*ts[t,1] for t in steps[4:])
# m.addConstrs(y[t-pt_delta] - y[t] >= path_len_y - M*ts[t,2] for t in steps[4:])
# m.addConstrs(y[t] - y[t-pt_delta] >= path_len_y - M*ts[t,3] for t in steps[4:])
# # m.addConstrs(ts[t,0] + ts[t,1] <= 1 for t in steps[pt_delta:])
# m.addConstrs(ts[t,0] + ts[t,1] + ts[t,2] + ts[t,3] <= 3 for t in steps[4:])

m.update()

obj = quicksum(f[t] for t in steps)
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()

# Print the variable values
# path = np.zeros(Np)
# print(m.getAttr('X', pos).values())
# for v in m.getVars():
#     if v.X != 0:
#         print("%s %f" % (v.varName, v.X))

# Plotting Code

path_x = m.getAttr('X', x).values()
path_y = m.getAttr('X', y).values()
print('Obj: %g' % obj.getValue())
path = list(zip(path_x, path_y))
print(path)
#path = list(zip(path_x.values(),path_y.values()))
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

plt.savefig('Pictures/mip_run_%s.png' % time.strftime("%Y%m%d-%H%M%S"))
plt.show()
