#!/home/mlfrantz/miniconda2/bin/python3.6
# /usr/bin/python

# Monte Carlo Tree Search algorithm

#This was inspired by:
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.

from math import *
import random
import time
from scipy.spatial.distance import euclidean as dist
import numpy as np

import sys, pdb, time, argparse, os, csv
import oyaml as yaml
import matplotlib.pyplot as plt
from sas_utils import World, Location

def normalize(data, index=0):
    # This function scales the data between 0-1. the 'index' variable is to select
    # a specific time frame to normalize to.
    x_min = np.min(data[:,:,index])
    x_max = np.max(data[:,:,index])
    return (data - x_min) / (x_max - x_min)

class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    def __init__(self, field, position, budget, path, end = None, direction_constr = '8_direction'):
        self.field = field # Scalar field
        self.pos = position # Position of robot, starts at the start imagine that
        self.end = end # Ending position
        self.budget = budget # How many hours are we planning for / or number of steps?
        self.path = path

        # Build the direction vectors for checking values
        self.dir_contr = direction_constr
        if self.dir_contr == '8_direction':
            # Check each of the 8 directions (N,S,E,W,NE,NW,SE,SW)
            self.directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
        elif self.dir_contr == 'nsew':
            self.directions = [(0,1), (0,-1), (1,0), (-1,0)] # N-S-E-W
        elif self.dir_contr == 'diag':
            self.directions = [(1,1), (-1,1), (1,-1), (-1,-1)] # Diag

    def Clone(self):
        """ Create a deep clone of this game state."""
        st = GameState(self.field, self.pos, self.budget, self.path, self.end, direction_constr=self.dir_contr)
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved."""
        # self.path.append(move) # Update path

        try:
            new_move = move[0][0]
        except:
            move = [move]
        # print("Do move:", move)
        self.pos = move # New position is the move
        self.budget -= 1
        # print(self.path)

    def GetMoves(self):
        """ Get all possible moves from this state."""
        # moves = []
        # next_int = self.pos
        # if next_int == self.goal:
        #     return []
        # else:
        #     #For our case the car can either go East, North, or South
        #     if next_int[0] + 1 < self.grid_size: #can move east
        #         moves.append((next_int[0]+1, next_int[1]))
        #     if next_int[1] + 1  < self.grid_size:# and (next_int[1] + 1 != self.last_pos[1]): #can move north
        #         moves.append((next_int[0], next_int[1]+1))
        #     if next_int[1] - 1  >= 0:# and (next_int[1] - 1 != self.last_pos[1]): #can move south
        #         moves.append((next_int[0], next_int[1]-1))
        #     for move in moves:
        #         if move == self.goal:
        #             return [self.goal]
        #         else:
        #             return moves

        # Check each of the directions
        moves = []
        # print(self.path)
        # print(self.budget)
        if self.budget <= 1: # If this path has exceeded our budget then we have no more moves
            return moves

        for i,d in enumerate(self.directions):
            # print(d[0], d[1])
            # print("pos: ", self.pos[0])
            # try/except prevents moving into not legal space
            # try:
            # if args.same_point:
            # Checks if next point has already been visited, if not, then add move
            try:
                move = [int(self.pos[0][0]) + int(d[0]), int(self.pos[0][1]) + int(d[1])]
            except TypeError:
                move =  [int(self.pos[0]) + int(d[0]), int(self.pos[1]) + int(d[1])]

            # print(move[0], move[1])
            if move not in self.path:
                # print("Move not in Path")
                # pdb.set_trace()
                if move[0] < 0 or move[0] >= self.field.shape[0] or move[1] < 0 or move[1] >= self.field.shape[1]:
                    # Makes sure we are in bounds
                    continue
                else:
                    # print(moves)
                    moves.append(move)
            else:
                continue
            # else:
            #     values[i] = self.field[self.path[-1][0] + d[0], self.path[-1][1] + d[1], 0]
            # except:
            #     continue
        return moves

    def GetResult(self, move):
        #if move is to the goal, end the rollout
        # temp_path = self.path[:] + [move]
        # if len(temp_path) >= self.budget:
        #     # exhausted our budget, end the rollout, return path score
        #     return self.field[move[0],move[1],0]#sum([self.field[p[0],p[1],0] for p in temp_path])
        # else:
        #     return 0.0
        return self.field[int(move[0][0]), int(move[0][1]), 0]

    def GetRandomMove(self):
        move = random.choice(self.GetMoves())
        # print("Random moves:", move)
        return [move]

    def __repr__(self):
        """ Don't need this - but good style.
        """
        s = "Current Position:" + str(self.pos) + " Goal:" + str(self.end)
        return s

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved."""
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s

def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone() #creates a deep copy of the state

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        # print(node.untriedMoves)
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove([m])
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(state.GetRandomMove())

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(state.pos)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    #else: print(rootnode.ChildrenToString())

    # return sorted(rootnode.childNodes, key = lambda c: c.wins)[-1].move # return the move that has the highest wins
    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that has the most visits

def UCTPlayGame(field, start, budget, end=None, direction_constr='8_direction'):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """

    state = GameState(field, start, budget, start, end, direction_constr)
    return_path = start
    # print(state.GetMoves())
    while (state.GetMoves() != []):
        # print(str(state))
        m = UCT(rootstate = state, itermax = 8*budget*100, verbose = False) # play with values for itermax and verbose = True
        # print("Best Move: " + str(m) + "\n")
        state.DoMove(m)
        state.path = state.path[:] + [m]
        return_path.append(m)
        # print(str(state))
    print(return_path)
    return return_path


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

    # Starting position contraint
    start = args.start_point
    if len(start) > 2:
        # More than one robot.
        start = [start[i:i + 2] for i in range(0, len(start), 2)]
    else:
        # One robot, extra list needed for nesting reasons.
        start = [start]

    startTime = time.time()

    # for s in steps[0]:
    #     # Check each of the directions
    #     if s == 0:
    #         path = start
    #         continue
    #     values = np.zeros(len(directions))
    #     for i,d in enumerate(directions):
    #         try:
    #             if args.same_point:
    #                 if [path[-1][0] + d[0], path[-1][1] + d[1]] not in path:
    #                     values[i] = field[path[-1][0] + d[0], path[-1][1] + d[1], 0]
    #                 else:
    #                     continue
    #             else:
    #                 values[i] = field[path[-1][0] + d[0], path[-1][1] + d[1], 0]
    #         except:
    #             continue
    #
    #     path.append([path[-1][0] + directions[np.argmax(values)][0], path[-1][1] + directions[np.argmax(values)][1]])

    path = UCTPlayGame(field, start, len(steps[0]), None, args.direction_constr)

    runTime = time.time() - startTime

    print(path)
    if args.gen_image:
        # Plotting Code
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
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

        print(sum([field[p[0],p[1],0] for p in path]))
        score_str = '_score_%f' % sum([field[p[0],p[1],0] for p in path])


        file_string = 'mcts_' + time.strftime("%Y%m%d-%H%M%S") + \
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
            dir_str = ''

        constraint_string = dir_str

        score_str = sum([field[p[0],p[1],0] for p in path])

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
                                'Algorithm': 'MCTS', \
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



if __name__ == "__main__":
    """ Play a single game to the end using UCT """
    main()
