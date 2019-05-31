# x and y values. Might consider changer to GRB.INTEGER
# x = {} m.addVars(range(Np), DX, name='x')
# y = {}
# f = {}
# for k in range(Np):
#     x[k] = m.addVar(lb=DX[0], ub=DX[-1], vtype=GRB.INTEGER, name='x%d' % k)
#     y[k] = m.addVar(lb=DY[0], ub=DY[-1], vtype=GRB.INTEGER, name='y%d' % k)
#     f[k] = m.addVar(vtype=GRB.CONTINUOUS, name='f%d' % k)
# x[0] = start[0]
# y[0] = start[1]
# f[0] = field[start[0], start[1]]

# lx = {} # Continuous variable for weight in x direction
# ly = {} # Continuous variable for weight in y direction
# lxy = {} # Continuous variable for weights in x*y matrix
# # Define a set of weights lx, ly, lxy
# for k in range(Np):
#     for i in range(numX):
#         lx[i,k] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='lx{0}{1}'.format(i,k))
#     for j in range(numY):
#         ly[j,k] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='ly{0}{1}'.format(j,k))
#     for i in range(numX):
#         for j in range(numY):
#             # i is column and j is row
#             lxy[i,j,k] = m.addVar(vtype=GRB.CONTINUOUS, name='lxy{0}{1}{2}'.format(i,j,k))
#
# b = {}
# for i in range(4):
#     b[i] = m.addVar(vtype=GRB.BINARY, name='b%d' % i)

# Add SOS2 constraints
#
# for k in range(Np):
#     m.addSOS(GRB.SOS_TYPE2, quicksum(DX[i]*lx[i,k] for i in range(numX)))
#     #m.addSOS(GRB.SOS_TYPE2, ly)

# Add objective
# m.setObjective(quicksum(f[k] for k in range(Np)), GRB.MAXIMIZE)
# m.update()
#
# for k in range(Np):
#     # Add row and column continuity
#     for i in range(numX):
#         m.addConstr(quicksum(lxy[i,j,k] for j in range(numY)) == lx[i])
#     for j in range(numY):
#         m.addConstr(quicksum(lxy[i,j,k] for i in range(numX)) == ly[j])
#     # # Add lx,ly sum to 1 constraint
#     m.addConstr(quicksum(lx[i,k] for i in range(numX)) == 1)
#     m.addConstr(quicksum(ly[j,k] for j in range(numY)) == 1)
#     m.addConstr(quicksum(DX[i]*lx[i] for i in range(numX)) == x[k])
#     m.addConstr(quicksum(DY[j]*ly[j] for j in range(numY)) == y[k])
#     m.addConstr(quicksum(field[i,j]*lxy[i,j] for i in range(numX) for j in range(numY)) == f[k])
#     # m.update()
#
# # for k in range(1,Np):
#     if k < Np-1:
#         m.addConstr(x[k] + b[0] - b[1] == x[k+1])
#         m.addConstr(b[0] + b[1] <= 1)
#         m.addConstr(y[k] + b[2] - b[3] == y[k+1])
#         m.addConstr(b[2] + b[3] <= 1)
#         m.addConstr(b[0] + b[1] + b[2] + b[3] >= 1)

    # m.update()

    # x[k] = quicksum(DX[i]*lx[i] for i in range(numX))
    # y[k] = quicksum(DY[j]*ly[j] for j in range(numY))
    # f[k] = quicksum(field[i,j]*lxy[i,j] for i in range(numX) for j in range(numY))
#m.optimize()

# print(x.values(), y.values(), f.values())
# for v in m.getVars():
#     print('%s %g' % (v.varName, v.x))

#print('Obj: %g' % obj.getValue())
