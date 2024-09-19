from template import Agent
import time, random
import numpy as np
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
from Azul.azul_model import AzulState





THINKTIME   = 0.9
NUM_PLAYERS = 2

def ScoreState(state: AzulState.AgentState) -> int:
    """
    ScoreState
    """
    score_inc = 0
    grid_state = deepcopy(state.grid_state)
    score = state.score

    # 1. Action tiles across from pattern lines to the wall grid
    for i in range(state.GRID_SIZE):
        # Is the pattern line full? If not it persists in its current
        # state into the next round.
        if state.lines_number[i] == i+1:
            tc = state.lines_tile[i]
            col = int(state.grid_scheme[i][tc])

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
            for j in range(col+1,state.GRID_SIZE,1):
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
            for j in range(i+1, state.GRID_SIZE, 1):
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
    for i in range(len(state.floor)):
        penalties += state.floor[i]*state.FLOOR_SCORES[i]
    
    
    # Agents cannot be assigned a negative score in any round.
    score_change = score_inc + penalties
    score += score_change
    # if score < 0:
    #     score = 0
    return score

class myAgent(Agent):
    def __init__(self,_id):
        self.id = _id # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS) # Agent stores an instance of GameRule, from which to obtain functions.

    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action):
        state = self.game_rule.generateSuccessor(state, action, self.id)
        
        # goal_reached = False #TODO: Students, how should agent check whether it reached goal or not
        goal_reached = not state.TilesRemaining()

        return goal_reached
        
    def SelectAction(self,actions,game_state):
        print("-"*25)
        gs = deepcopy(game_state)
        original_score = gs.agents[self.id].score
        scores = []
        for a in actions:
            t_gs = deepcopy(game_state)
            goal = self.DoAction(t_gs, a)
            new_score = ScoreState(t_gs.agents[self.id])
            if goal and new_score < 0:
                new_score = 0
            scores.append(new_score-original_score)
        return actions[np.argmax(scores)]
        # return random.choice(actions)

