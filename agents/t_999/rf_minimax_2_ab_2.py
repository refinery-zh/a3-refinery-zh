# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    04/01/2021
# Purpose: Implements an example breadth-first search agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time, random
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
import Azul.azul_utils as utils
from copy import deepcopy
from collections import deque
import numpy as np

THINKTIME = 0.9
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


def GetScore(id, game_state: AzulState):
    my_gs = game_state.agents[id]
    op_gs = game_state.agents[1 - id]
    return ScoreState(my_gs) - ScoreState(op_gs)


def ScoreState(agent_state: AzulState.AgentState):
    score_inc = 0
    grid_state = deepcopy(agent_state.grid_state)
    number_of = deepcopy(agent_state.number_of)
    score = agent_state.score

    # 1. Action tiles across from pattern lines to the wall grid
    for i in range(agent_state.GRID_SIZE):
        # Is the pattern line full? If not it persists in its current
        # state into the next round.
        if agent_state.lines_number[i] == i + 1:
            tc = agent_state.lines_tile[i]
            col = int(agent_state.grid_scheme[i][tc])

            # Tile will be placed at position (i,col) in grid
            grid_state[i][col] = 1
            number_of[tc] += 1

            # count the number of tiles in a continguous line
            # above, below, to the left and right of the placed tile.
            above = 0
            for j in range(col - 1, -1, -1):
                val = grid_state[i][j]
                above += val
                if val == 0:
                    break
            below = 0
            for j in range(col + 1, agent_state.GRID_SIZE, 1):
                val = grid_state[i][j]
                below += val
                if val == 0:
                    break
            left = 0
            for j in range(i - 1, -1, -1):
                val = grid_state[j][col]
                left += val
                if val == 0:
                    break
            right = 0
            for j in range(i + 1, agent_state.GRID_SIZE, 1):
                val = grid_state[j][col]
                right += val
                if val == 0:
                    break

            # If the tile sits in a contiguous vertical line of
            # tiles in the grid, it is worth 1*the number of tiles
            # in this line (including itstate).
            if above > 0 or below > 0:
                score_inc += 1 + above + below

            # In addition to the vertical score, the tile is worth
            # an additional H points where H is the length of the
            # horizontal contiguous line in which it sits.
            if left > 0 or right > 0:
                score_inc += 1 + left + right

            # If the tile is not next to any already placed tiles
            # on the grid, it is worth 1 point.
            if above == 0 and below == 0 and left == 0 and right == 0:
                score_inc += 1

    # Score penalties for tiles in floor line
    penalties = 0
    for i in range(len(agent_state.floor)):
        penalties += agent_state.floor[i] * agent_state.FLOOR_SCORES[i]

    rows = 0
    for i in range(agent_state.GRID_SIZE):
        allin = True
        for j in range(agent_state.GRID_SIZE):
            if grid_state[i][j] == 0:
                allin = False
                break
        if allin:
            rows += 1
    cols = 0
    for i in range(agent_state.GRID_SIZE):
        allin = True
        for j in range(agent_state.GRID_SIZE):
            if grid_state[j][i] == 0:
                allin = False
                break
        if allin:
            cols += 1
    sets = 0
    for tile in utils.Tile:
        if number_of[tile] == agent_state.GRID_SIZE:
            sets += 1

    bonus = (rows * agent_state.ROW_BONUS) + (cols * agent_state.COL_BONUS) + (sets * agent_state.SET_BONUS)
    # Agents cannot be assigned a negative score in any round.
    score_change = score_inc + penalties
    score += score_change
    if score < 0:
        score = 0
    score += bonus
    return score


# Defines this agent.
class myAgent:
    def __init__(self, _id):
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(
            NUM_PLAYERS
        )  # Agent stores an instance of GameRule, from which to obtain functions.
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.

    def dfs_explore(self, actions, game_state:AzulState, depth, id, parents, start_time):
        if depth == 0:
            assert True
            return None, GetScore(self.id, game_state)
        scores = []
        result = None
        assert actions != []
        for a in actions:
            if time.time() - start_time > THINKTIME:
                break
            gs = deepcopy(game_state)
            gs = self.game_rule.generateSuccessor(gs, a, id)
            if a == "ENDROUND":
                score = GetScore(self.id, game_state)
            else:
                new_actions = self.game_rule.getLegalActions(gs, 1 - id)
                _, score = self.dfs_explore(new_actions, gs, depth - 1, 1 - id, result, start_time)
            scores.append(score)
            if parents == "root":
                continue
            if self.id == id:
                if result == None:
                    result = score
                else:
                    result = max(score, result)
                if parents != None and parents <= result:
                    return np.argmax(scores), result
            else:
                if result == None:
                    result = score
                else:
                    result = min(score, result)
                if parents != None and parents >= result:
                    return np.argmin(scores), result
        return np.argmax(scores), result

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, game_state):
        start_time = time.time()
        depth = int(2*np.log(50) / (np.log(len(actions)+1)+0.1))
        # print(depth)
        if depth < 1:
            depth = 1
        idx, _ = self.dfs_explore(actions, game_state, depth, self.id, "root", start_time)
        return actions[idx]


# END FILE -----------------------------------------------------------------------------------------------------------#
