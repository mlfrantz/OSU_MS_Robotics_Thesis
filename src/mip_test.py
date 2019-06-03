#!/usr/bin/python

import sys, pdb, time, argparse
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

def main():

    parser = argparse.ArgumentParser(description='Parser for MIP testing')
    parser.add_argument(
        '-i', '--infile_path',
        nargs='?',
        type=str,
        default="/home/mlfrantz/Documents/MIP_Research/mip_research/test_fields/test_field_2.csv",
        help='Input file that represents the world',
        )
    parser.add_argument(
        '-o', '--outfile_path',
        nargs='?',
        type=str,
        default="/home/mlfrantz/Documents/MIP_Research/mip_research/Pictures/",
        help='Directory where pictures are stored',
        )
    parser.add_argument(
        '-g','--gradient',
        action='store_true',
        help='By adding this flag you will compute the gradient of the input field.',
        )
    parser.add_argument(
        '-n', '--path_lenth',
        nargs='?',
        type=int,
        default=5,
        help='Length of the path to be planned in (units).',
        )
    parser.add_argument(
        '-s', '--start_point',
        nargs=2,
        type=int,
        default=(0,0),
        help='Starting point for planning purposes, returns list [x,y].',
        )
    parser.add_argument(
        '-e', '--end_point',
        nargs=2,
        type=int,
        default=[],
        help='Ending point for planning purposes, returns list [x,y].',
        )
    parser.add_argument(
        '-t', '--time_limit',
        nargs='?',
        type=float,
        default=0.0,
        help='Time limit in seconds you want to stop the simulation. Default lets it run until completion.',
        )
    parser.add_argument(
        '-d', '--direction_constr',
        nargs='?',
        type=str,
        default='8_direction',
        help='Sets the direction constraint. Default allows it to move in any of the 8 directions each move. \
        "nsew" only lets it move north-south-east-west. \
        "diag" only lets it move diagonally (NW,NE,SW,SE).',
        )
    parser.add_argument(
        '--same_point',
        action='store_false',
        help='By default it will not allow a point to be visited twice in the same planning period.',
        )
    parser.add_argument(
        '--anti_curl',
        action='store_true',
        help='By default it will not consider the anti-curling constraints.',
        )
    parser.add_argument(
        '--force_curl',
        action='store_true',
        help='By default it will not consider the force-curling constraints.',
        )
    parser.add_argument(
        '--straight_line',
        # action='store_true',
        # help='By default it will not consider the straight line constraints.',
        nargs=2,
        type=int,
        default=[],
        help='Sets how long you want each straight to be, returns list [x,y].',
        )
    parser.add_argument(
        '--gen_image',
        action='store_true',
        help='Set to true if you want the image to be saved to file.',
        )

    args = parser.parse_args()

    Np = args.path_lenth # Path lenth Np
    steps = range(Np)

    # Problem data, matrix transposed to allow for proper x,y coordinates to be mapped wih i,j
    field = np.genfromtxt(args.infile_path, delimiter=',', dtype=float).transpose()

    if args.gradient:
        grad_field = np.gradient(field)
        mag_grad_field = np.sqrt(grad_field[0]**2 + grad_field[1]**2)

    DX = np.arange(field.shape[0]) # Integer values for range of X coordinates
    DY = np.arange(field.shape[1]) # Integer values for range of Y coordinates

    m = Model()
    if args.time_limit > 0:
        m.Params.TIME_LIMIT = args.time_limit
    # m.Params.MIPGap = 0.01

    # Add variables
    x = m.addVars(steps, lb=DX[0], ub=DX[-1], vtype=GRB.CONTINUOUS, name='x')
    y = m.addVars(steps, lb=DY[0], ub=DY[-1], vtype=GRB.CONTINUOUS, name='y')

    # If we are taking the gradient of the field
    if args.gradient:
        f = m.addVars(steps,lb=np.min(mag_grad_field), ub=np.max(mag_grad_field), vtype=GRB.CONTINUOUS, name='f')
    else:
        f = m.addVars(steps, lb=np.min(field), ub=np.max(field), vtype=GRB.CONTINUOUS, name='f')

    lx = m.addVars(steps, DX, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='lx')
    ly = m.addVars(steps, DY, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='ly')
    lxy = m.addVars(steps, DX, DY, vtype=GRB.CONTINUOUS, name='lxy')
    for t in steps:
        m.addSOS(GRB.SOS_TYPE2, [lx[t,i] for i in DX])
        m.addSOS(GRB.SOS_TYPE2, [ly[t,j] for j in DY])

    # Add main constraints

    # Starting position contraint
    start = args.start_point
    m.addConstr((x[steps[0]] == start[0]), name="Initial x")
    m.addConstr((y[steps[0]] == start[1]), name="Initial y")

    if args.gradient:
        m.addConstr((f[steps[0]] == mag_grad_field[start[0], start[1]]), name="Initial f")
    else:
        m.addConstr((f[steps[0]] == field[start[0], start[1]]), name="Initial f")

    # Optional end position constraint. Could implement additional position constaint
    if len(args.end_point) > 0 :
        end = args.end_point
        m.addConstr((x[steps[-1]] == end[0]), name="End x")
        m.addConstr((y[steps[-1]] == end[1]), name="End y")
        m.addConstr((f[steps[-1]] == field[end[0], end[1]]), name="End f")

    for t in steps:
        m.addConstrs(quicksum(lxy[t, i, j] for j in DY) == lx[t, i] for i in DX)
        m.addConstrs(quicksum(lxy[t, i, j] for i in DX) == ly[t, j] for j in DY)

    m.addConstrs((quicksum(lx[t, i] for i in DX) == 1 for t in steps))
    m.addConstrs((quicksum(ly[t, j] for j in DY) == 1 for t in steps))

    m.addConstrs((quicksum(DX[i]*lx[t,i] for i in DX) == x[t] for t in steps))
    m.addConstrs((quicksum(DY[j]*ly[t,j] for j in DY) == y[t] for t in steps))

    if args.gradient:
        m.addConstrs((quicksum(mag_grad_field[i,j]*lxy[t,i,j] for i in DX for j in DY) == f[t] for t in steps))
    else:
        m.addConstrs((quicksum(field[i,j]*lxy[t,i,j] for i in DX for j in DY) == f[t] for t in steps))

    # Primary Motion constraints
    # Binary variables for motion constraints
    b_range = range(4)
    b = m.addVars(steps, b_range, vtype=GRB.BINARY, name='b')
    m.addConstrs(x[t-1] + b[t,0] - b[t,1] == x[t] for t in steps[1:])
    m.addConstrs(b[t,0] + b[t,1] <= 1 for t in steps[1:])
    m.addConstrs(y[t-1] + b[t,2] - b[t,3] == y[t] for t in steps[1:])
    m.addConstrs(b[t,2] + b[t,3] <= 1 for t in steps[1:])

    # Add option constraints

    # Direction constraints PCA/Eignevector/Cardinal constraints
    if args.direction_constr == '8_direction':
        m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] >= 1 for t in steps[1:])
    elif args.direction_constr == 'nsew':
        m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] == 1 for t in steps[1:]) #PCA N-S-E-W
    elif args.direction_constr == 'diag':
        m.addConstrs(b[t,0] + b[t,1] + b[t,2] + b[t,3] == 2 for t in steps[1:]) #PCA Diag

    # Constraint to prevent going through the same point twice
    if args.same_point:
        M = 100
        for t in steps[1:]:
            t1 = m.addVars(t, range(4), vtype=GRB.BINARY, name='t%d'% t)
            for s in range(t):
                m.addConstr(x[t]-x[s] >= 0.1 - M*t1[s,0])
                m.addConstr(x[s]-x[t] >= 0.1 - M*t1[s,1])
                m.addConstr(y[t]-y[s] >= 0.1 - M*t1[s,2])
                m.addConstr(y[s]-y[t] >= 0.1 - M*t1[s,3])
                m.addConstr(t1[s,0] + t1[s,1] + t1[s,2] + t1[s,3] <= 3)

    # Anti-Curling constraints go from [3,...,Np] and enforce ani-curling (this one works!)
    if args.anti_curl:
        t_range = range(4)
        t1 = m.addVars(steps, t_range, vtype=GRB.BINARY, name='t1')
        delta = 2
        t_delta = 3
        M = 100
        m.addConstrs(x[t-2] - x[t] >= delta - M*t1[t,0] for t in steps[t_delta:])
        m.addConstrs(x[t] - x[t-2] >= delta - M*t1[t,1] for t in steps[t_delta:])
        m.addConstrs(y[t-2] - y[t] >= delta - M*t1[t,2] for t in steps[t_delta:])
        m.addConstrs(y[t] - y[t-2] >= delta - M*t1[t,3] for t in steps[t_delta:])
        m.addConstrs(t1[t,0] + t1[t,1] + t1[t,2] + t1[t,3] <= 3 for t in steps[t_delta:])

    # Curling constraints go from [2,...,Np] and enforce curing (Questionable functionality)
    if args.force_curl:
        delta = 2
        t_delta = 3
        m.addConstrs(x[t-t_delta] - x[t] <= delta for t in steps[t_delta:])
        m.addConstrs(x[t] - x[t-t_delta] <= delta for t in steps[t_delta:])
        m.addConstrs(y[t-t_delta] - y[t] <= delta for t in steps[t_delta:])
        m.addConstrs(y[t] - y[t-t_delta] <= delta for t in steps[t_delta:])

    # Straight path constraints (Questionable functionality)
    if len(args.straight_line) > 0:
        # Working better
        path_len_x = args.straight_line[0]
        path_len_y = args.straight_line[1]
        delta = 4
        M = 100
        ts = m.addVars(steps, range(4), vtype=GRB.BINARY, name='t1')
        m.addConstrs(x[t-delta] - x[t] >= path_len_x - M*ts[t,0] for t in steps[delta:])
        m.addConstrs(x[t] - x[t-delta] >= path_len_x - M*ts[t,1] for t in steps[delta:])
        m.addConstrs(y[t-delta] - y[t] >= path_len_y - M*ts[t,2] for t in steps[delta:])
        m.addConstrs(y[t] - y[t-delta] >= path_len_y - M*ts[t,3] for t in steps[delta:])
        m.addConstrs(ts[t,0] + ts[t,1] + ts[t,2] + ts[t,3] <= 3 for t in steps[delta:])

    # End constraint updates, seting objective
    m.update()
    obj = quicksum(f[t] for t in steps)
    m.setObjective(obj, GRB.MAXIMIZE)

    # Run the optimizer
    m.optimize()

    # Print the variable values
    # path = np.zeros(Np)
    # print(m.getAttr('X', pos).values())
    # for v in m.getVars():
    #     if v.X != 0:
    #         print("%s %f" % (v.varName, v.X))

    if args.gen_image:
        # Plotting Code
        path_x = m.getAttr('X', x).values()
        path_y = m.getAttr('X', y).values()
        # print('Obj: %g' % obj.getValue())
        path = list(zip(path_x, path_y))
        # print(path)
        #path = list(zip(path_x.values(),path_y.values()))
        if args.gradient:
            plt.imshow(mag_grad_field.transpose(), interpolation='gaussian', cmap= 'gnuplot')
        else:
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

        path_len_str = '_pathLen_%d' % Np

        if args.gradient:
            grad_str = '_gradient'
        else:
            grad_str = ''

        if args.time_limit > 0:
            time_lim_str = '_timeLim_%d' % args.time_limit
        else:
            time_lim_str = ''

        if args.direction_constr == 'nsew':
            dir_str = '_%s' % args.direction_constr
        elif args.direction_constr == 'diag':
            dir_str = '_%s' % args.direction_constr
        else:
            dir_str = ''

        if args.anti_curl:
            anti_curl_str = '_antiCurl'
        else:
            anti_curl_str = ''

        if args.force_curl:
            force_curl_str = '_forceCurl'
        else:
            force_curl_str = ''

        if len(args.straight_line) > 0:
            straight_line_str = '_straight_%d_%d' % (args.straight_line[0], args.straight_line[1])
        else:
            straight_line_str = ''

        obj = m.getObjective()
        score_str = '_score_%d' % obj.getValue()

        file_string = 'mip_run_' + time.strftime("%Y%m%d-%H%M%S") + \
                                                                    grad_str + \
                                                                    path_len_str + \
                                                                    time_lim_str + \
                                                                    dir_str + anti_curl_str + \
                                                                    force_curl_str + \
                                                                    straight_line_str + \
                                                                    score_str + \
                                                                    '.png'

        print(file_string)
        plt.savefig(args.outfile_path + file_string)
        plt.show()

if __name__ == '__main__':
    main()
