from copy import deepcopy
from math import sqrt, log
from Azul.azul_model import AzulGameRule as GameRule
import random
import time
import numpy as np

THINKTIME = 0.9
NUM_PLAYERS = 2

def ScoreState(state):
    score_inc = 0
    grid_state = deepcopy(state.grid_state)

    # 1. Action tiles across from pattern lines to the wall grid
    for i in range(state.GRID_SIZE):
        # Is the pattern line full? If not it persists in its current
        # state into the next round.
        ori_score = score_inc
        if state.lines_number[i] == i+1:
            tc = state.lines_tile[i]
            col = int(state.grid_scheme[i][tc])
            # print(tc, col)
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

            # print(score_inc - ori_score)

    # Score penalties for tiles in floor line
    penalties = 0
    for i in range(len(state.floor)):
        penalties += state.floor[i]*state.FLOOR_SCORES[i]
    # print(f'From ScoreState: {score_inc}, {penalties}')
    # Agents cannot be assigned a negative score in any round.
    score_change = score_inc + penalties
    return score_change

import time

class myAgent:
    def __init__(self, _id):
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule.
        self.best_action = None

    def minimax(self, state, depth, maximizingPlayer, alpha, beta, start_time, think_time, current_agent_id):
        # Check if time limit has been reached
        if time.time() - start_time > think_time:
            return None  # Timeout
        
        if depth == 0 or not state.TilesRemaining():
            # Return the heuristic value (state score)
            print(f'Agent {current_agent_id} reached depth limit, and the score is {ScoreState(state.agents[current_agent_id])}')
            return ScoreState(state.agents[self.id]) - ScoreState(state.agents[(self.id + 1) % NUM_PLAYERS])

        legal_actions = self.game_rule.getLegalActions(state, current_agent_id)

        print(f'Current depth: {depth}')

        if maximizingPlayer:
            print(f'Agent {current_agent_id} is maximizing...')
            maxEval = float('-inf')
            for action in legal_actions:
                next_state = deepcopy(state)
                next_state = self.game_rule.generateSuccessor(next_state, action, current_agent_id)
                eval = self.minimax(next_state, depth - 1, False, alpha, beta, start_time, think_time, (current_agent_id + 1) % NUM_PLAYERS)

                if eval is None:  # If time runs out, stop further evaluation
                    return None

                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return maxEval
        else:
            print(f'Agent {current_agent_id} is minimizing...')
            minEval = float('inf')
                
            for action in legal_actions:
                next_state = deepcopy(state)
                next_state = self.game_rule.generateSuccessor(next_state, action, current_agent_id)
                eval = self.minimax(next_state, depth - 1, True, alpha, beta, start_time, think_time, (current_agent_id + 1) % NUM_PLAYERS)

                if eval is None:  # If time runs out, stop further evaluation
                    return None

                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return minEval

    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        think_time = THINKTIME  # Defined global think time limit
        depth = 2
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        print(f'Agent {self.id} is thinking...')

        while time.time() - start_time < think_time:
            current_best_action = None
            current_best_value = float('-inf')

            # Iterate through each legal action for the current depth
            for action in actions:
                next_state = deepcopy(rootstate)
                next_state = self.game_rule.generateSuccessor(next_state, action, self.id)
                
                # Call the minimax function for the opponent (minimizing player)
                value = self.minimax(next_state, depth, False, alpha, beta, start_time, think_time, (self.id + 1) % NUM_PLAYERS)

                if value is None:  # Timeout occurred
                    break

                # Keep track of the best move found at this depth
                if value > current_best_value:
                    current_best_value = value
                    current_best_action = action

            # If we have a valid action for this depth, update the best action
            if current_best_action is not None:
                best_action = current_best_action
                best_value = current_best_value

            # Dynamically increase depth if time allows
            depth += 1

        # exit()
        # Return the best action found within the time limit
        return best_action if best_action is not None else random.choice(actions)