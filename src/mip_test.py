#!/home/mlfrantz/miniconda2/bin/python3.6

import sys, pdb, time, argparse, os, csv
import oyaml as yaml
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
from sas_utils import World, Location

def normalize(data, index=0):
    # This function scales the data between 0-1. the 'index' variable is to select
    # a specific time frame to normalize to.
    x_min = np.min(data[:,:,index])
    x_max = np.max(data[:,:,index])
    return (data - x_min) / (x_max - x_min)

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
        '-r', '--robots',
        nargs='*',
        type=str,
        default='glider1',
        help='List of robots to plan for. Must be in the robots.yaml file.',
        )
    parser.add_argument(
        '--robots_cfg',
        nargs='?',
        type=str,
        default='cfg/robots.yaml',
        help='Configuration file of robots availalbe for planning.',
        )
    parser.add_argument(
        '--sim_cfg',
        nargs='?',
        type=str,
        default='cfg/sim.yaml',
        help='Simulation-specific configuration file name.',
        )
    parser.add_argument(
        '-n', '--planning_time',
        nargs='?',
        type=float,
        default=5,
        help='Length of the path to be planned in (units).',
        )
    parser.add_argument(
        '-s', '--start_point',
        nargs='*',
        type=int,
        default=(0,0),
        help='Starting points for robots for planning purposes, returns list [x0,y0,x1,y1,...,xN,yN] for 1...N robots.',
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
        '--sync',
        nargs='?',
        type=str,
        default='None',
        help='Sets the direction of the Synchronization constraint. Default is no Synchronization. \
        "ns" goes from north to south. \
        "sn" goes from south to north. \
        "ew" goes from east to west. \
        "we" goes from west to east. \
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
        '-a', '--rect_area',
        nargs=4,
        type=int,
        default=[],
        help='Coordinates for recangular are to restrict planning to, returns list [x1, x2, y1, y2].',
        )
    parser.add_argument(
        '-c', '--collision_rad',
        nargs='?',
        type=float,
        default=0.0,
        help='Collision Radius between robots.',
        )
    parser.add_argument(
        '--gen_image',
        action='store_true',
        help='Set to true if you want the image to be saved to file.',
        )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Will load ROMS maps by default, otherwise loads a test map.',
        )
    parser.add_argument(
        '--experiment_name',
        nargs='?',
        type=str,
        default="Test Experiment",
        help='Name of the Experiement you are running',
        )

    args = parser.parse_args()

    # Path lenth in time (hours).
    Np = args.planning_time

    # Load the map from either ROMS data or test file
    if not args.test:
        # ROMS map
        # Loading Simulation-Specific Parameters
        with open(os.path.expandvars(args.sim_cfg),'rb') as f:
            yaml_sim = yaml.load(f.read())

        wd = World.roms(
            datafile_path=yaml_sim['roms_file'],
            xlen        = yaml_sim['sim_world']['width'],
            ylen        = yaml_sim['sim_world']['height'],
            center      = Location(xlon=yaml_sim['sim_world']['center_longitude'], ylat=yaml_sim['sim_world']['center_latitude']),
            feature     = yaml_sim['science_variable'],
            resolution  = (yaml_sim['sim_world']['resolution'],yaml_sim['sim_world']['resolution']),
            )

        # This is the scalar_field in a static word.
        # The '0' is the first time step and goes up to some max time
        # field = np.copy(wd.scalar_field[:,:,0])
        field = np.copy(wd.scalar_field)

        norm_field = normalize(field)
        field = normalize(field) # This will normailze the field between 0-1


        # Example of an obstacle, make the value very low in desired area
        # field[int(len(field)/4):int(3*len(field)/4),int(len(field)/4):int(3*len(field)/4)] = -100

        field_resolution = (yaml_sim['sim_world']['resolution'],yaml_sim['sim_world']['resolution'])

    else:
        # Problem data, matrix transposed to allow for proper x,y coordinates to be mapped wih i,j
        field = np.genfromtxt(args.infile_path, delimiter=',', dtype=float).transpose()
        field_resolution = (1,1)

    if args.gradient:
        grad_field = np.gradient(field)
        mag_grad_field = np.sqrt(grad_field[0]**2 + grad_field[1]**2)

    # Load the robots.yaml Configuration file.
    with open(os.path.expandvars(args.robots_cfg),'rb') as f:
        yaml_mission = yaml.load(f.read())

    # Get the speed of each robot that we are planning for.
    steps = []
    colors = []
    for key,value in [(k,v) for k,v in yaml_mission.items() if k in args.robots]:
        # Number of 1Km steps the planner can plan for.
        # The expresseion solves for the number of waypoints so a +1 is needed for range.
        # For instance, a glider going 0.4m/s would travel 1.44Km in 1 hour so it needs at least 2 waypoints, start and end.
        plan_range = int(np.round(value['vel']*Np*60*60*0.001*(1/min(field_resolution))))+1
        if plan_range > 0:
            steps.append(range(plan_range))
        else:
            steps.append(range(2))
        colors.append(value['color'])

    temp_len = [len(s) for s in steps]
    steps = [steps[np.argmax(temp_len)]]*len(steps) # This makes everything operate at the same time
    velocity_correction = [t/max(temp_len) for t in temp_len] # To account for time difference between arriving to waypoints

    # Make time correction for map forward propagation
    max_steps = max([len(s) for s in steps])
    field_delta = int(max_steps/Np)
    t_step = 0
    k_step = 0
    field_time_steps = []
    for i in range(max_steps):
        field_time_steps.append(t_step)
        k_step += 1
        if k_step == field_delta:
            k_step = 0
            t_step += 1

    # Number of robots we are planning for.
    robots = range(len(args.robots))

    DX = np.arange(field.shape[0]) # Integer values for range of X coordinates
    DY = np.arange(field.shape[1]) # Integer values for range of Y coordinates

    m = Model()
    if args.time_limit > 0:
        m.Params.TIME_LIMIT = args.time_limit
    # m.Params.MIPGap = 0.01

    # Add variables
    pairs = tuplelist([(r,s) for r in robots for s in steps[r]])

    x = m.addVars(pairs, lb=DX[0], ub=DX[-1], vtype=GRB.CONTINUOUS, name='x')
    y = m.addVars(pairs, lb=DY[0], ub=DY[-1], vtype=GRB.CONTINUOUS, name='y')

    # If we are taking the gradient of the field
    if args.gradient:
        f = m.addVars(pairs ,lb=np.min(mag_grad_field), ub=np.max(mag_grad_field), vtype=GRB.CONTINUOUS, name='f')
    else:
        f = m.addVars(pairs, lb=np.min(field), ub=np.max(field), vtype=GRB.CONTINUOUS, name='f')

    lx = m.addVars(pairs, DX, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='lx')
    ly = m.addVars(pairs, DY, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='ly')
    lxy = m.addVars(pairs, DX, DY, vtype=GRB.CONTINUOUS, name='lxy')

    for r in robots:
        for t in steps[r]:
            m.addSOS(GRB.SOS_TYPE2, [lx[r,t,i] for i in DX])
            m.addSOS(GRB.SOS_TYPE2, [ly[r,t,j] for j in DY])

    # Add main constraints

    # Starting position contraint
    start = args.start_point
    if len(start) > 2:
        # More than one robot.
        start = [start[i:i + 2] for i in range(0, len(start), 2)]
    else:
        # One robot, extra list needed for nesting reasons.
        start = [start]

    m.addConstrs((x[r, steps[r][0]] == start[r][0] for r in robots), name="Initial x")
    m.addConstrs((y[r, steps[r][0]] == start[r][1] for r in robots), name="Initial y")

    if args.gradient:
        m.addConstrs((f[r, steps[r][0]] == mag_grad_field[start[r][0], start[r][1]] for r in robots), name="Initial f")
    else:
        m.addConstrs((f[r, steps[r][0]] == field[start[r][0], start[r][1], 0] for r in robots), name="Initial f")

    # Optional end position constraint. Could implement additional position constaint
    if len(args.end_point) > 0 :
        end = args.end_point
        m.addConstr((x[r, steps[r][-1]] == end[0] for r in robots), name="End x")
        m.addConstr((y[r, steps[r][-1]] == end[1] for r in robots), name="End y")
        m.addConstr((f[r, steps[r][-1]] == field[end[0], end[1], -1] for r in robots), name="End f")

    for r in robots:
        for t in steps[r]:
            m.addConstrs(quicksum(lxy[r, t, i, j] for j in DY) == lx[r, t, i] for i in DX)
            m.addConstrs(quicksum(lxy[r, t, i, j] for i in DX) == ly[r, t, j] for j in DY)

    m.addConstrs((quicksum(lx[r, t, i] for i in DX) == 1 for r in robots for t in steps[r]))
    m.addConstrs((quicksum(ly[r, t, j] for j in DY) == 1 for r in robots for t in steps[r]))

    m.addConstrs((quicksum(DX[i]*lx[r,t,i] for i in DX) == x[r,t] for r in robots for t in steps[r]))
    m.addConstrs((quicksum(DY[j]*ly[r,t,j] for j in DY) == y[r,t] for r in robots for t in steps[r]))

    if args.gradient:
        m.addConstrs((quicksum(mag_grad_field[i,j]*lxy[r,t,i,j] for i in DX for j in DY) == f[r,t] for r in robots for t in steps[r]))
    else:
        m.addConstrs((quicksum(field[i,j,field_time_steps[t]]*lxy[r,t,i,j] for i in DX for j in DY) == f[r,t] for r in robots for t in steps[r]))

    # Primary Motion constraints
    # Binary variables for motion constraints
    b_range = range(4)
    b = m.addVars(pairs, b_range, vtype=GRB.BINARY, name='b%d' % t)

    # There is something to this but it doesnt work as is
    block = 1
    for r in robots:
        v = velocity_correction[r]
        for t in steps[r][1:]:
            # pdb.set_trace()
            m.addConstr(x[r,t-1] + v*block*b[r,t,0] - v*block*b[r,t,1] == x[r,t])
            m.addConstr(b[r,t,0] + b[r,t,1] <= 1)
            m.addConstr(y[r,t-1] + v*b[r,t,2] - v*b[r,t,3] == y[r,t])
            m.addConstr(b[r,t,2] + b[r,t,3] <= 1)

    # Add option constraints

    # Direction constraints PCA/Eignevector/Cardinal constraints
    if args.direction_constr == '8_direction':
        m.addConstrs(b[r,t,0] + b[r,t,1] + b[r,t,2] + b[r,t,3] >= 1 for r in robots for t in steps[r][1:])
    elif args.direction_constr == 'nsew':
        m.addConstrs(b[r,t,0] + b[r,t,1] + b[r,t,2] + b[r,t,3] == 1 for r in robots for t in steps[r][1:]) #PCA N-S-E-W
    elif args.direction_constr == 'diag':
        m.addConstrs(b[r,t,0] + b[r,t,1] + b[r,t,2] + b[r,t,3] == 2 for r in robots for t in steps[r][1:]) #PCA Diag

    # Constraint to prevent going through the same point twice
    if args.same_point:
        M = 100
        for r in robots:
            for i,t in enumerate(steps[r][1:]):
                t1 = m.addVars(t, range(4), vtype=GRB.BINARY, name='t%d'% t)
                for j,s in enumerate(steps[r][:i+1]):
                    m.addConstr(x[r,t]-x[r,s] >= 0.1 - M*t1[j,0])
                    m.addConstr(x[r,s]-x[r,t] >= 0.1 - M*t1[j,1])
                    m.addConstr(y[r,t]-y[r,s] >= 0.1 - M*t1[j,2])
                    m.addConstr(y[r,s]-y[r,t] >= 0.1 - M*t1[j,3])
                    m.addConstr(t1[j,0] + t1[j,1] + t1[j,2] + t1[j,3] <= 3)

    # Synchronization Constraint: Specific path or 8 direction [NS, EW, NE-SW, NW-SE]
    if args.sync == 'ns':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t-1] - x[r,t] == 0)
                m.addConstr(y[r,t] - y[r,t-1]  == v)
    elif args.sync == 'sn':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t-1] - x[r,t] == 0)
                m.addConstr(y[r,t-1] - y[r,t] == v)
    elif args.sync == 'ew':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t-1] - x[r,t] == v)
                m.addConstr(y[r,t-1] - y[r,t] == 0)
    elif args.sync == 'we':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t] - x[r,t-1] == v)
                m.addConstr(y[r,t-1] - y[r,t] == 0)
    elif args.sync == 'nwse':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t] - x[r,t-1] == v)
                m.addConstr(y[r,t] - y[r,t-1] == v)
    elif args.sync == 'nesw':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t-1] - x[r,t] == v)
                m.addConstr(y[r,t] - y[r,t-1] == v)
    elif args.sync == 'senw':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t-1] - x[r,t] == v)
                m.addConstr(y[r,t-1] - y[r,t] == v)
    elif args.sync == 'swne':
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t] - x[r,t-1] == v)
                m.addConstr(y[r,t-1] - y[r,t] == v)
    elif args.sync == 'path':
        path = [(0,5), (1,4), (2,3), (3,2), (4,1), (5,0), (6,1), (7,2), (8,3), (9,4), (10,5), (9,6), (8,7), \
        (7,8), (6,9), (5,10), (4,9), (3,8), (2,7), (1,6), (0,5)]
        path_len = np.sum(np.abs(np.subtract(path[:-1],path[1:])))
        for r in robots:
            v = velocity_correction[r]
            for t in steps[r][1:]:
                m.addConstr(x[r,t] - x[r,t-1] == path[t][0] - path[t-1][0])
                m.addConstr(y[r,t] - y[r,t-1] == path[t][1] - path[t-1][1])


    # Constraint to prevent colliding with other robots.
    if args.collision_rad > 0:
        M = 100
        t2 = m.addVars(len(robots), len(robots), range(4), vtype=GRB.BINARY, name='t2')
        for r in robots[1:]:
            for s in range(r):
                m.addConstrs(x[r,t]-x[s,t] >= args.collision_rad - M*t2[r,s,0] for t in steps[s][1:])
                m.addConstrs(x[s,t]-x[r,t] >= args.collision_rad - M*t2[r,s,1] for t in steps[s][1:])
                m.addConstrs(y[r,t]-y[s,t] >= args.collision_rad - M*t2[r,s,2] for t in steps[s][1:])
                m.addConstrs(y[s,t]-y[r,t] >= args.collision_rad - M*t2[r,s,3] for t in steps[s][1:])
                m.addConstr(t2[r,s,0] + t2[r,s,1] + t2[r,s,2] + t2[r,s,3] <= 3)

    # Anti-Curling constraints go from [3,...,Np] and enforce ani-curling (this one works!)
    if args.anti_curl:
        for r in robots:
            v = velocity_correction[r]
            t_range = range(4)
            t1 = m.addVars(steps[r], t_range, vtype=GRB.BINARY, name='t1')
            delta = 2
            t_delta = 3
            M = 100
            for t in steps[r][t_delta:]:
                m.addConstr(x[r,t-2] - x[r,t] >= v*delta - M*t1[t,0] )
                m.addConstr(x[r,t] - x[r,t-2] >= v*delta - M*t1[t,1])
                m.addConstr(y[r,t-2] - y[r,t] >= v*delta - M*t1[t,2])
                m.addConstr(y[r,t] - y[r,t-2] >= v*delta - M*t1[t,3])
                m.addConstr(t1[t,0] + t1[t,1] + t1[t,2] + t1[t,3] <= 3)

    # Curling constraints go from [2,...,Np] and enforce curing (Questionable functionality)
    if args.force_curl:
        for r in robots:
            v = velocity_correction[r]
            delta = 2
            t_delta = 3
            m.addConstrs(x[r,t-t_delta] - x[r,t] <= v*delta for t in steps[r][t_delta:])
            m.addConstrs(x[r,t] - x[r,t-t_delta] <= v*delta for t in steps[r][t_delta:])
            m.addConstrs(y[r,t-t_delta] - y[r,t] <= v*delta for t in steps[r][t_delta:])
            m.addConstrs(y[r,t] - y[r,t-t_delta] <= v*delta for t in steps[r][t_delta:])

    # Straight path constraints (Questionable functionality)
    if len(args.straight_line) > 0:
        # Now we need to correct our previous velocity_correction by making sure the edges are the lengths of the edges are all equal.
        path_len_x = args.straight_line[0]
        path_len_y = args.straight_line[1]
        delta = max(path_len_x,path_len_y)
        M = 1000
        for r in robots:
            for v in velocity_correction:
                inv_v = 1/v
                if inv_v > 1:
                    tv = m.addVars(steps[r][delta:], range(4), vtype=GRB.BINARY, name='t1')
                    for t in steps[r][delta:]:
                        m.addConstr(x[r,t-delta] - x[r,t] >= path_len_x*v - M*tv[t,0] )
                        m.addConstr(x[r,t] - x[r,t-delta] >= path_len_x*v - M*tv[t,1] )
                        m.addConstr(y[r,t-delta] - y[r,t] >= path_len_y*v - M*tv[t,2] )
                        m.addConstr(y[r,t] - y[r,t-delta] >= path_len_y*v - M*tv[t,3] )
                        m.addConstr(tv[t,0] + tv[t,1] + tv[t,2] + tv[t,3] == 3 )
                else:
                    # Working better
                    ts = m.addVars(steps[r][delta:], range(4), vtype=GRB.BINARY, name='ts')
                    for t in steps[r][delta:]:
                        m.addConstr(x[r,t-delta] - x[r,t] >= path_len_x - M*ts[t,0])
                        m.addConstr(x[r,t] - x[r,t-delta] >= path_len_x - M*ts[t,1])
                        m.addConstr(y[r,t-delta] - y[r,t] >= path_len_y - M*ts[t,2])
                        m.addConstr(y[r,t] - y[r,t-delta] >= path_len_y - M*ts[t,3])
                        m.addConstr(ts[t,0] + ts[t,1] + ts[t,2] + ts[t,3] == 3 )

    # Area Constraint (Start with square, expand to more complex shapes)
    if len(args.rect_area) > 0:
        for r in robots:
            area = args.rect_area
            m.addConstrs(x[r,t] >= area[0] for t in steps[r][1:])
            m.addConstrs(x[r,t] <= area[1] for t in steps[r][1:])
            m.addConstrs(y[r,t] >= area[2] for t in steps[r][1:])
            m.addConstrs(y[r,t] <= area[3] for t in steps[r][1:])

    # End constraint updates, seting objective
    m.update()
    obj = quicksum(f[r,t] for r in robots for t in steps[r])
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
        _paths = list(zip(path_x, path_y))
        paths = []
        for r in robots:
            paths.append(_paths[0:len(steps[r])])
            _paths = _paths[len(steps[r]):]

        print(paths)

        if args.gradient:
            # plt.imshow(mag_grad_field.transpose())#, interpolation='gaussian', cmap= 'gnuplot')
            if not args.test:
                plt.imshow(wd.scalar_field[:,:,0].transpose(), interpolation='gaussian', cmap= 'gnuplot')
                plt.xticks(np.arange(0,len(wd.lon_ticks), (1/min(field_resolution))), np.around(wd.lon_ticks[0::int(1/min(field_resolution))], 2))
                plt.yticks(np.arange(0,len(wd.lat_ticks), (1/min(field_resolution))), np.around(wd.lat_ticks[0::int(1/min(field_resolution))], 2))
                plt.xlabel('Longitude', fontsize=20)
                plt.ylabel('Latitude', fontsize=20)
                plt.text(1.25, 0.5, str(yaml_sim['science_variable']),{'fontsize':20}, horizontalalignment='left', verticalalignment='center', rotation=90, clip_on=False, transform=plt.gca().transAxes)
            else:
                plt.imshow(field.transpose(), interpolation='gaussian', cmap= 'gnuplot')
        else:
            if not args.test:
                plt.imshow(norm_field[:,:,0].transpose(), interpolation='gaussian', cmap= 'gnuplot')
                plt.xticks(np.arange(0,len(wd.lon_ticks), (1/min(field_resolution))), np.around(wd.lon_ticks[0::int(1/min(field_resolution))], 2))
                plt.yticks(np.arange(0,len(wd.lat_ticks), (1/min(field_resolution))), np.around(wd.lat_ticks[0::int(1/min(field_resolution))], 2))
                plt.xlabel('Longitude', fontsize=20)
                plt.ylabel('Latitude', fontsize=20)
                plt.text(1.25, 0.5, "normalized " + str(yaml_sim['science_variable']),{'fontsize':20}, horizontalalignment='left', verticalalignment='center', rotation=90, clip_on=False, transform=plt.gca().transAxes)
            else:
                plt.imshow(field.transpose(), interpolation='gaussian', cmap= 'gnuplot')

        plt.colorbar()
        for i,path in enumerate(paths):
            path_x = [x for x,y in path]
            path_y = [y for x,y in path]
            plt.plot(path_x, path_y, color=colors[i], linewidth=2.0)
            plt.plot(path[0][0], path[0][1], color='g', marker='o')
            plt.plot(path[-1][0], path[-1][1], color='r', marker='o')

        points = []
        for path in paths:
            for i, point in enumerate(path):
                points.append(points)
                if point in path and point not in point:
                    plt.annotate(i+1, (path[i][0], path[i][1]))

        robots_str = '_robots_%d' % len(robots)

        path_len_str = '_pathLen_%d' % len(steps[0])

        if len(args.end_point) > 0 :
            end_point_str = '_end%d%d' % (args.end_point[0], args.end_point[1])
        else:
            end_point_str = ''

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

        if args.collision_rad > 0:
            collision_str = '_collRad_%d' % args.collision_rad
        else:
            collision_str = ''

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

        if len(args.rect_area) > 0:
            area = args.rect_area
            rect_area_str = '_rect_x%d_%dy%d_%d' % (area[0], area[1], area[2], area[3])
        else:
            rect_area_str = ''

        obj = m.getObjective()
        score_str = '_score_%d' % obj.getValue()

        file_string = 'mip_run_' + time.strftime("%Y%m%d-%H%M%S") + \
                                                                    robots_str + \
                                                                    path_len_str + \
                                                                    end_point_str + \
                                                                    collision_str + \
                                                                    grad_str + \
                                                                    time_lim_str + \
                                                                    dir_str + anti_curl_str + \
                                                                    force_curl_str + \
                                                                    straight_line_str + \
                                                                    score_str + \
                                                                    rect_area_str + \
                                                                    '.png'

        print(file_string)
        plt.savefig(args.outfile_path + file_string)
        plt.show()
    else:
        filename = args.outfile_path
        check_empty = os.path.exists(filename)

        if args.direction_constr == 'nsew':
            dir_str = '_%s' % args.direction_constr
        elif args.direction_constr == 'diag':
            dir_str = '_%s' % args.direction_constr
        else:
            dir_str = ''

        if args.collision_rad > 0:
            collision_str = '_collRad_%d' % args.collision_rad
        else:
            collision_str = ''

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

        if len(args.rect_area) > 0:
            area = args.rect_area
            rect_area_str = '_rect_x%d_%dy%d_%d' % (area[0], area[1], area[2], area[3])
        else:
            rect_area_str = ''

        constraint_string = collision_str + \
                            dir_str + anti_curl_str + \
                            force_curl_str + \
                            straight_line_str + \
                            rect_area_str

        obj = m.getObjective()
        score_str = obj.getValue()

        with open(filename, 'a', newline='') as csvfile:
            fieldnames = [  'Experiment', \
                            'Algorithm', \
                            'Map', \
                            'Map Center', \
                            'Map Resolution', \
                            'Start Point', \
                            'End Point', \
                            'Score', \
                            'Run Time (sec)', \
                            'Budget (hours)', \
                            'Number of Robots', \
                            'Constraints']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not check_empty:
                print("File is empty")
                writer.writeheader()

            writer.writerow({   'Experiment': args.experiment_name, \
                                'Algorithm': 'MIP', \
                                'Map': str(yaml_sim['roms_file']), \
                                'Map Center': Location(xlon=yaml_sim['sim_world']['center_longitude'], ylat=yaml_sim['sim_world']['center_latitude']).__str__(), \
                                'Map Resolution': (yaml_sim['sim_world']['resolution'],yaml_sim['sim_world']['resolution']), \
                                'Start Point': args.start_point, \
                                'End Point': args.end_point if len(args.end_point) > 0 else 'NA' , \
                                'Score': score_str, \
                                'Run Time (sec)': m.Runtime, \
                                'Budget (hours)': args.planning_time, \
                                'Number of Robots': len(args.robots), \
                                'Constraints': constraint_string})



if __name__ == '__main__':
    main()
