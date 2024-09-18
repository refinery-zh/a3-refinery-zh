# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    04/01/2021
# Purpose: Implements an example breadth-first search agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time, random
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from copy import deepcopy
from collections import deque
import numpy as np
import heapq
import Azul.azul_utils as utils

THINKTIME   = 0.9
NUM_PLAYERS = 2

# FUNCTIONS ----------------------------------------------------------------------------------------------------------#

class Node:
    def __init__(self, agent_id, state, action, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.score_difference = None
        self.agent_id = agent_id
        self.action = action
        self.expanded = False

class MinimaxTree:
    def __init__(self, _id):
        self.game_rule = GameRule(NUM_PLAYERS)
        self.id = _id
        self.root = None

    def get_score(self, game_state:AzulState):
        my_gs = game_state.agents[self.id]
        op_gs = game_state.agents[1-self.id]
        return self.score_state(my_gs) - self.score_state(op_gs)

    def score_state(self, agent_state:AzulState.AgentState):
        score_inc = 0
        grid_state = deepcopy(agent_state.grid_state)
        score = agent_state.score

        # 1. Action tiles across from pattern lines to the wall grid
        for i in range(agent_state.GRID_SIZE):
            # Is the pattern line full? If not it persists in its current
            # state into the next round.
            if agent_state.lines_number[i] == i+1:
                tc = agent_state.lines_tile[i]
                col = int(agent_state.grid_scheme[i][tc])

                # Tile will be placed at position (i,col) in grid
                grid_state[i][col] = 1

                # count the number of tiles in a continguous line
                # above, below, to the left and right of the placed tile.
                above = 0
                for j in range(col-1, -1, -1):
                    val = grid_state[i][j]
                    above += val
                    if val == 0:
                        break
                below = 0
                for j in range(col+1,agent_state.GRID_SIZE,1):
                    val = grid_state[i][j]
                    below +=  val
                    if val == 0:
                        break
                left = 0
                for j in range(i-1, -1, -1):
                    val = grid_state[j][col]
                    left += val
                    if val == 0:
                        break
                right = 0
                for j in range(i+1, agent_state.GRID_SIZE, 1):
                    val = grid_state[j][col]
                    right += val
                    if val == 0:
                        break

                # If the tile sits in a contiguous vertical line of 
                # tiles in the grid, it is worth 1*the number of tiles
                # in this line (including itstate).
                if above > 0 or below > 0:
                    score_inc += (1 + above + below)

                # In addition to the vertical score, the tile is worth
                # an additional H points where H is the length of the 
                # horizontal contiguous line in which it sits.
                if left > 0 or right > 0:
                    score_inc += (1 + left + right)

                # If the tile is not next to any already placed tiles
                # on the grid, it is worth 1 point.                
                if above == 0 and below == 0 and left == 0 and right == 0:
                    score_inc += 1

        # Score penalties for tiles in floor line
        penalties = 0
        for i in range(len(agent_state.floor)):
            penalties += agent_state.floor[i]*agent_state.FLOOR_SCORES[i]
        
        
        # Agents cannot be assigned a negative score in any round.
        score_change = score_inc + penalties
        score += score_change
        if score < 0:
            score = 0
        return score

    def dfs_explore(self, node:Node, depth_remain, start_time):
        if time.time() - start_time > THINKTIME:
            if node.score_difference == None:
                node.score_difference = self.get_score(node.state)
            return node.score_difference
        if depth_remain == 0:
            node.score_difference = self.get_score(node.state)
            return node.score_difference
        else:
            if node.expanded == False:
                new_actions = self.game_rule.getLegalActions(node.state, node.agent_id)
                if new_actions == []:
                    node.score_difference = self.get_score(node.state)
                    return node.score_difference
                for action in new_actions:
                    next_state = deepcopy(node.state) 
                    new_state = self.game_rule.generateSuccessor(next_state, action, node.agent_id)
                    new_node = Node(1-node.agent_id, new_state, action, node)
                    node.children.append(new_node)
            
            # assert type(node) == Node
            for child in node.children:
                score = self.dfs_explore(child, depth_remain-1, start_time)
                if node.agent_id == self.id:
                    if node.score_difference == None:
                        node.score_difference = score
                    elif node.parent != None and node.parent.score_difference != None:
                        if node.score_difference >= node.parent.score_difference:
                            return node.score_difference
                    else:
                        node.score_difference = max(node.score_difference, score)
                else:
                    if node.score_difference == None:
                        node.score_difference = score
                    elif node.parent != None and node.parent.score_difference != None:
                        if node.score_difference <= node.parent.score_difference:
                            return node.score_difference
                    else:
                        node.score_difference = min(node.score_difference, score)
        return node.score_difference
    
    def to_node(self, node:Node):
        tmp = None
        for child in self.root.children:
            if node == child:
                tmp = child
                break
        self.root = tmp
            
    def update_root(self, game_state):
        if self.root == None:
            self.root = Node(self.id, game_state, None, None)
        else:
            tmp = None
            for child in self.root.children:
                if child.state == game_state:
                    tmp = child
                    break
            if tmp != None:
                self.root = tmp
            else:
                self.root = Node(self.id, game_state, None, None)

    def delete(self, node:Node):
        if node.children == []:
            del node
        else:
            for c in node.children:
                self.delete(c)

# Defines this agent.
class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS) # Agent stores an instance of GameRule, from which to obtain functions.
        self.minimax_tree = MinimaxTree(self.id)
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.
    
    def get_best_node(self, root:Node):
        node = None
        score = -1000000
        # assert type(node) == Node
        for child in root.children:
            if child.score_difference > score:
                node = child
                score = child.score_difference
        return node
    
    def num_of_steps(self, game_state:AzulState):
        # print("*"*50)
        tiles = np.zeros([6,5])
        # print("*"*50)
        for i in range(len(game_state.factories)):
            factory = game_state.factories[i]
            for tile in utils.Tile:
                if factory.tiles[tile] != 0:
                    tiles[i][tile] = 1
        # print("*"*50)
        for tile in utils.Tile:
            if game_state.centre_pool.tiles[tile] != 0:
                tiles[-1][tile] = 1
        max_step = np.sum(tiles)
        # print("*"*50)
        for tile in utils.Tile:
            if tiles[-1][tile] == 1:
                tiles[:][tile] = 0
        min_step = np.sum(tiles)
        # print("*"*50)
        print(min_step, max_step)
        return min_step, max_step

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, game_state):
        start_time = time.time()
        self.minimax_tree.update_root(game_state)
        min_step, max_step = self.num_of_steps(game_state)
        if max_step < 8:
            self.minimax_tree.dfs_explore(self.minimax_tree.root, 3, start_time)
        else:
            self.minimax_tree.dfs_explore(self.minimax_tree.root, 2, start_time)
        best_node = self.get_best_node(self.minimax_tree.root)
        best_action = best_node.action
        self.minimax_tree.to_node(best_node)

        return best_action
    
# END FILE -----------------------------------------------------------------------------------------------------------#