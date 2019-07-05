#!/usr/bin/python

# Monte Carlo Tree Search algorithm

#This was inspired by:
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.

#Edge wait based on expected network congestion, use for score calculation

from math import *
import random
import time
from scipy.spatial.distance import euclidean as dist
import numpy as np

class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    def __init__(self, field, start, position, budget, path, end = None, direction_constr = None):
            self.field = field # Scalar field
            self.pos = start # Position of robot, starts at the start imagine that
            self.end = end # Ending position
            self.budget = budget # How many hours are we planning for
            self.path = [path]

            # Build the direction vectors for checking values
            if direction_constr == 'None':
                # Check each of the 8 directions (N,S,E,W,NE,NW,SE,SW)
                self.directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
            elif direction_constr == 'nsew':
                self.directions = [(0,1), (0,-1), (1,0), (-1,0)] # N-S-E-W
            elif direction_constr == 'diag':
                self.directions = [(1,1), (-1,1), (1,-1), (-1,-1)] # Diag

    def Clone(self):
        """ Create a deep clone of this game state."""
        st = GameState(self.field, self.pos, self.budget, self.end, self.path, direction_constr=self.directions)
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved."""
        self.path.append(move) # Update path
        self.pos = move # New position is the move

    def GetMoves(self):
        """ Get all possible moves from this state."""
        moves = []
        next_int = self.pos
        if next_int == self.goal:
            return []
        else:
            #For our case the car can either go East, North, or South
            if next_int[0] + 1 < self.grid_size: #can move east
                moves.append((next_int[0]+1, next_int[1]))
            if next_int[1] + 1  < self.grid_size:# and (next_int[1] + 1 != self.last_pos[1]): #can move north
                moves.append((next_int[0], next_int[1]+1))
            if next_int[1] - 1  >= 0:# and (next_int[1] - 1 != self.last_pos[1]): #can move south
                moves.append((next_int[0], next_int[1]-1))
            for move in moves:
                if move == self.goal:
                    return [self.goal]
                else:
                    return moves

    def GetResult(self, move):
        #if move is to the goal, end the rollout
        temp_path = self.path[:] + move
        if len(temp_path) == self.budget:
            # exhausted our budget, end the rollout, return path score
            return sum([self.field[p[0],p[1],0] for p in temp_path])
        else:
            return 0.0

    def GetRandomMove(self):
        moves = self.GetMoves()
        return random.choice(moves)

    def __repr__(self):
        """ Don't need this - but good style.
        """
        s = "Current Position:" + str(self.pos) + " Goal:" + str(self.goal)
        return s

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
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
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
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

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

def UCTPlayGame(car, GRID_SIZE, other_cars = None):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """
    state = GameState(car, GRID_SIZE, other_cars)
    while (state.GetMoves() != []):
        #print(str(state))
        m = UCT(rootstate = state, itermax = 120, verbose = False) # play with values for itermax and verbose = True
        #print("Best Move: " + str(m) + "\n")
        state.DoMove(m)
    return state.path


if __name__ == "__main__":
    """ Play a single game to the end using UCT """
    UCTPlayGame()
