# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    04/01/2021
# Purpose: Implements an example breadth-first search agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time, random
import numpy as np
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from copy import deepcopy
from queue import Queue

THINKTIME   = 0.9
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#

class Node:
    def __init__(self, layer, agent_id, state, action, parent=None):
        self.layer = layer
        self.agent_id = agent_id
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.score_difference = None
        self.expanded = False
        self.goal = False
        self.dropped = False

class MinimaxTree:
    def __init__(self, _id):
        self.game_rule = GameRule(NUM_PLAYERS)
        self.id = _id
        self.root = None
        self.current_layer = 0
        self.queue = Queue()

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
    
    def expand(self, node:Node):
        layer = node.layer
        node.expanded = True
        if not node.state.TilesRemaining():
            node.goal = True
        else:
            new_actions = self.game_rule.getLegalActions(node.state, node.agent_id)
            for action in new_actions:
                next_state = deepcopy(node.state)
                # new_state = self.game_rule.generateSuccessor(next_state, action, node.agent_id)
                self.game_rule.generateSuccessor(next_state, action, node.agent_id)
                new_node = Node(layer+1, 1-node.agent_id, next_state, action, node)
                node.children.append(new_node)
                self.queue.put(new_node)
    
    def bfs_explore(self, start_time):
        cnt = 0
        while not self.queue.empty() and time.time()-start_time < THINKTIME:
            node = self.queue.get()
            # print(self.root.layer, node.layer, self.current_layer)
            # print(node, node.dropped)
            if node.dropped:
                continue
            cnt += 1
            if not node.expanded:
                self.expand(node)
            if node.layer > self.current_layer:
                if self.current_layer > self.root.layer:
                    # print("\n"*20)
                    # print("update")
                    # start_time = time.time()
                    self.reset(self.current_layer, self.root)
                    self.update(self.current_layer, self.root)
                    # for c in self.root.children:
                    #     print(c.score_difference)
                    # input()
                self.current_layer += 1
        print(cnt)

    def reset(self, layer, node:Node):
        if node.layer == layer:
            node.score_difference = None
        else:
            for child in node.children:
                self.reset(layer, child)

    def update(self, layer, node:Node):
        if node.layer == layer:
            node.score_difference = self.get_score(node.state)
            return node.score_difference
        else:
            for child in node.children:
                score = self.update(layer, child)
                if node.agent_id == self.id:
                    if node.score_difference == None:
                        node.score_difference = score
                    # elif node.parent != None and node.parent.score_difference != None:
                    #     if node.score_difference >= node.parent.score_difference:
                    #         return node.score_difference
                    else:
                        node.score_difference = max(node.score_difference, score)
                else:
                    if node.score_difference == None:
                        node.score_difference = score
                    # elif node.parent != None and node.parent.score_difference != None:
                    #     if node.score_difference <= node.parent.score_difference:
                    #         return node.score_difference
                    else:
                        node.score_difference = min(node.score_difference, score)
        return node.score_difference
                
    def to_node(self, node:Node):
        tmp = None
        for child in self.root.children:
            if node == child:
                tmp = child
            else:
                self.delete(child)
        # del self.root
        self.root.dropped = True
        self.root = tmp
            
    def update_root(self, game_state):
        if self.root == None:
            self.root = Node(0, self.id, game_state, None, None)
            self.current_layer = 0
            self.queue = Queue()
            self.queue.put(self.root)
        else:
            tmp = None
            for child in self.root.children:
                if self.same_state(child.state, game_state):
                    tmp = child
                else:
                    self.delete(child)
            self.root.dropped = True
            if tmp != None:
                self.root = tmp
            else:
                self.root = Node(0, self.id, game_state, None, None)
                self.current_layer = 0
                self.queue = Queue()
                self.queue.put(self.root)
    
    def same_state(self, current:AzulState, target:AzulState):
        for i in range(len(current.factories)):
            for tile in range(5):
                if current.factories[i].tiles[tile] != target.factories[i].tiles[tile]:
                    return False
        for tile in range(5):
            if current.centre_pool.tiles[tile] != target.centre_pool.tiles[tile]:
                return False
        if current.first_agent_taken != target.first_agent_taken:
            return False
        return True

    def delete(self, node:Node):
        node.dropped = True
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
        if root.children[-1].score_difference == None:
            return None
        else:
            for child in root.children:
                # assert type(child.score_difference) == int
                if child.score_difference > score:
                    node = child
                    score = child.score_difference
        return node

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, game_state):
        print(len(actions))
        start_time = time.time()
        self.minimax_tree.update_root(game_state)
        assert type(self.minimax_tree.root) == Node
        self.minimax_tree.bfs_explore(start_time)
        best_node = self.get_best_node(self.minimax_tree.root)
        if best_node != None:
            best_action = best_node.action
            self.minimax_tree.to_node(best_node)
        else:
            print("GREEDY")
            scores = []
            for a in actions:
                t_gs = deepcopy(game_state)
                t_gs = self.game_rule.generateSuccessor(t_gs, a, self.id)
                new_score = self.minimax_tree.score_state(t_gs.agents[self.id])
                scores.append(new_score)
            best_action = actions[np.argmax(scores)]
        # return random.choice(actions)
        return best_action
# END FILE -----------------------------------------------------------------------------------------------------------#