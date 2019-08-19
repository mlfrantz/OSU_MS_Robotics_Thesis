#!/home/mlfrantz/miniconda2/bin/python3.6

# This is a greedy one step lookahead for comparison to my MIP implementation

import sys, pdb, time, argparse, os, csv
import random
import oyaml as yaml
import numpy as np
import matplotlib.pyplot as plt
from sas_utils import World, Location

def normalize(data, index=0):
    # This function scales the data between 0-1. the 'index' variable is to select
    # a specific time frame to normalize to.
    x_min = np.min(data[:,:,index])
    x_max = np.max(data[:,:,index])
    return (data - x_min) / (x_max - x_min)

def bilinear_interpolation(point, field):
        # Solution courtesy of Raymond Hettinger
        # https://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
    '''Interpolate (x,y) from point values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    x = point[0]
    y = point[1]
    try:
        points = sorted(corners(point,field))               # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1) + 0.0)
    except TypeError:
        print("Failed, no solution.")

def corners(point,field):
    corners = []

    pointX = point[0]
    pointY = point[1]

    for p in [pointX, pointY]:
        if float(p) % 1 != 0.0:
            # Means point is not an integer
            upper = np.ceil(p)
            lower = np.floor(p)
        else:
            if float(p) == 0.0:
                lower = p
                upper = 1
            elif p == field.shape[0]-1:
                upper = field.shape[0]-1
                lower = upper - 1
            else:
                upper = p
                lower = p + 1
        corners.append(int(lower))
        corners.append(int(upper))
    try:
        corners = [ (corners[0], corners[2], field[corners[0], corners[2], 0]), \
                    (corners[0], corners[3], field[corners[0], corners[3], 0]), \
                    (corners[1], corners[2], field[corners[1], corners[2], 0]), \
                    (corners[1], corners[3], field[corners[1], corners[3], 0])]
    except:
        # pdb.set_trace()
        print("Failed, no solution.")
    return corners

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
        '--same_point',
        action='store_false',
        help='By default it will not allow a point to be visited twice in the same planning period.',
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

        fieldSavePath = '/home/mlfrantz/Documents/MIP_Research/mip_research/cfg/normal_field_{}_{}.npy'.format(str(abs(yaml_sim['sim_world']['center_longitude'])),yaml_sim['sim_world']['center_latitude'])

        try:
            field = np.load(fieldSavePath)
            norm_field = np.load(fieldSavePath)
            print("Loaded Map Successfully")
        except IOError:

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

            # fieldSavePath = '/home/mlfrantz/Documents/MIP_Research/mip_research/cfg/normal_field.npy'
            np.save(fieldSavePath, field)

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

    # Starting position contraint
    start = args.start_point
    if len(start) > 2:
        # More than one robot.
        start = [start[i:i + 2] for i in range(0, len(start), 2)]
    else:
        # One robot, extra list needed for nesting reasons.
        start = [start]

    # Greedy one step look ahead

    # Build the direction vectors for checking values
    if args.direction_constr == '8_direction':
        # Check each of the 8 directions (N,S,E,W,NE,NW,SE,SW)
        directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    elif args.direction_constr == 'nsew':
        directions = [(0,1), (0,-1), (1,0), (-1,0)] # N-S-E-W
    elif args.direction_constr == 'diag':
        directions = [(1,1), (-1,1), (1,-1), (-1,-1)] # Diag

    startTime = time.time()


    paths = []
    for r in robots:
        for s in steps[r]:
            # Check each of the directions
            if s == 0:
                path = [start[r]]
                continue
            values = np.zeros(len(directions))

            for i,d in enumerate(directions):
                try:
                    if args.same_point:
                        move = [path[-1][0] + velocity_correction[r]*d[0], path[-1][1] + velocity_correction[r]*d[1]]
                        if move[0] >= 0 and move[0] <= field.shape[0]-1 and move[1] >= 0 and move[1] <= field.shape[1]-1:
                            # print(move,move[0], move[1],move[0] >= 0 and move[0] < field.shape[0] and move[1] >= 0 and move[1] < field.shape[1])
                            # Makes sure we are in
                            if [round(move[0],3),round(move[1],3)] not in [[round(p[0],3),round(p[1],3)] for p in path]:
                                # print(move)
                                values[i] = bilinear_interpolation(move, field)
                                # values[i] = field[path[-1][0] + d[0], path[-1][1] + d[1], 0]
                            else:
                                continue
                        else:
                            # print(move[0], move[1],move[0] >= 0 and move[0] < field.shape[0] and move[1] >= 0 and move[1] < field.shape[1])
                            # Makes sure we are in
                            continue
                    else:
                        move = [path[-1][0] + velocity_correction[r]*d[0], path[-1][1] + velocity_correction[r]*d[1]]
                        if move[0] >= 0 and move[0] <= field.shape[0]-1 and move[1] >= 0 and move[1] <= field.shape[1]-1:
                            values[i] = bilinear_interpolation(move, field)
                except:
                    continue
            # print(values)
            chosen = random.choice(list(enumerate(values)))
            count = 0
            while chosen[1] == 0 and count < 25:
                chosen = random.choice(list(enumerate(values)))
                count += 1

            new_point = [path[-1][0] + velocity_correction[r]*directions[chosen[0]][0], \
                         path[-1][1] + velocity_correction[r]*directions[chosen[0]][1]]
            # print(new_point, values, np.argmax(values), directions[np.argmax(values)])
            path.append(new_point)
        paths.append(path)
    # print(paths)
    runTime = time.time() - startTime

    if args.gen_image:
        wd = World.roms(
            datafile_path=yaml_sim['roms_file'],
            xlen        = yaml_sim['sim_world']['width'],
            ylen        = yaml_sim['sim_world']['height'],
            center      = Location(xlon=yaml_sim['sim_world']['center_longitude'], ylat=yaml_sim['sim_world']['center_latitude']),
            feature     = yaml_sim['science_variable'],
            resolution  = (yaml_sim['sim_world']['resolution'],yaml_sim['sim_world']['resolution']),
            )

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
            dir_str = 'none'

        # print(sum([field[p[0],p[1],0] for p in path]))

        try:
            score_str = '_score_%f' % sum([bilinear_interpolation(p, field) for path in paths for p in path])
        except TypeError:
            score_str = '_no_solution'

        file_string = 'random_' + time.strftime("%Y%m%d-%H%M%S") + \
                                                                    robots_str + \
                                                                    path_len_str + \
                                                                    end_point_str + \
                                                                    grad_str + \
                                                                    time_lim_str + \
                                                                    dir_str + \
                                                                    score_str + \
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
            dir_str = 'none'

        constraint_string = dir_str

        if not args.same_point:
            constraint_string = "same_point"

        try:
            score_str = sum([bilinear_interpolation(p, field) for path in paths for p in path])
        except TypeError:
            score_str = 0#'_no_solution'

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
                                'Algorithm': 'Random', \
                                'Map': str(yaml_sim['roms_file']), \
                                'Map Center': Location(xlon=yaml_sim['sim_world']['center_longitude'], ylat=yaml_sim['sim_world']['center_latitude']).__str__(), \
                                'Map Resolution': (yaml_sim['sim_world']['resolution'],yaml_sim['sim_world']['resolution']), \
                                'Start Point': args.start_point, \
                                'End Point': args.end_point if len(args.end_point) > 0 else 'NA' , \
                                'Score': score_str, \
                                'Run Time (sec)': runTime, \
                                'Budget (hours)': args.planning_time, \
                                'Number of Robots': len(args.robots), \
                                'Constraints': constraint_string})


if __name__ == '__main__':
    main()
