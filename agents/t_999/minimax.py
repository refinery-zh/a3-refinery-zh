from copy import deepcopy
from math import sqrt, log
from Azul.azul_model import AzulGameRule as GameRule
import random
import time
import numpy as np

THINKTIME = 0.9
NUM_PLAYERS = 2

def ScoreState(state):
    """
    Description:
    Get the score of the current state given that the round has not ended.

    Args:
    state (AzulState): The current state of the game.

    Returns:
    score_change (int): The score of the current state.
    """

    score_inc = 0
    # calculate the score for tiles moved to grid_state
    grid_state = deepcopy(state.grid_state)

    # even though the round has not ended
    # move tiles from pattern lines to wall grid
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
    # return negative score to show penalties
    # should not be bounded by 0
    score_change = score_inc + penalties
    return score_change

class myAgent:
    def __init__(self, _id):
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)

    def minimax(self, state, depth, maximizingPlayer, alpha, beta, start_time, current_agent_id):
        """
        Description:
        Minimax algorithm with alpha-beta pruning to find the best action for the agent.

        Args:
        state (AzulState): The current state of the game.
        depth (int): The depth of the search tree.
        maximizingPlayer (bool): Whether the current agent is maximizing or minimizing.
        alpha (float): The best value that the maximizing agent currently can guarantee at that level or above.
        beta (float): The best value that the minimizing agent currently can guarantee at that level or above.
        start_time (float): The time when the search started.
        current_agent_id (int): The id of the current agent.

        Returns:
        score (float): The score of the current state.
        """
        # return None if timeout
        if time.time() - start_time > THINKTIME:
            return None
        
        if depth == 0 or not state.TilesRemaining():
            print(f'Agent {current_agent_id} reached depth limit, and the score is {ScoreState(state.agents[current_agent_id])}')
            # the difference between the scores of our agent and the opponent
            return ScoreState(state.agents[self.id]) - ScoreState(state.agents[(self.id + 1) % NUM_PLAYERS])

        legal_actions = self.game_rule.getLegalActions(state, current_agent_id)

        print(f'Current depth: {depth}')

        if maximizingPlayer:
            print(f'Agent {current_agent_id} is maximizing...')
            maxScore = float('-inf')
            for action in legal_actions:
                next_state = deepcopy(state)
                # generate successors with current agent id
                next_state = self.game_rule.generateSuccessor(next_state, action, current_agent_id)
                score = self.minimax(next_state, depth - 1, False, alpha, beta, start_time, (current_agent_id + 1) % NUM_PLAYERS)

                if score is None:
                    if maxScore == float('-inf'):
                        return None
                    else:
                        return maxScore

                maxScore = max(maxScore, score)
                alpha = max(alpha, score)
                # alpha-beta pruning for maximizing
                if beta <= alpha:
                    break
            return maxScore
        else:
            print(f'Agent {current_agent_id} is minimizing...')
            minScore = float('inf')
                
            for action in legal_actions:
                next_state = deepcopy(state)
                # generate successors with current agent id
                next_state = self.game_rule.generateSuccessor(next_state, action, current_agent_id)
                score = self.minimax(next_state, depth - 1, True, alpha, beta, start_time, (current_agent_id + 1) % NUM_PLAYERS)

                if score is None:
                    if minScore == float('inf'):
                        return None
                    else:
                        return minScore

                minScore = min(minScore, score)
                beta = min(beta, score)
                # alpha-beta pruning for minimizing
                if beta <= alpha:
                    break
            return minScore

    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        depth = 2
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        print(f'Agent {self.id} is thinking...')

        while time.time() - start_time < THINKTIME:

            # at most search to depth 4
            # to prevent timeout
            # at the beginning of each round, the number of actions is large
            # which limits the depth
            # most of the time, the minimax stops at depth 3
            if (depth == 4):
                break

            best_action_current_depth = None
            # for our agent, we want to maximize the score
            best_value_current_depth = float('-inf')

            for action in actions:
                next_state = deepcopy(rootstate)
                next_state = self.game_rule.generateSuccessor(next_state, action, self.id)
                print(f'Action: {action}')
                # minimize the opponent's score
                # the id should be the opponent's id
                # because current agent already has successors generated
                value = self.minimax(next_state, depth, False, alpha, beta, start_time, (self.id + 1) % NUM_PLAYERS)

                # no value returned when timeout
                if value is None:
                    break

                if value > best_value_current_depth:
                    best_value_current_depth = value
                    best_action_current_depth = action

            # update best action and value if current depth is better
            if best_action_current_depth is not None:
                # coz there may be timeout
                # the value returned may be suboptimal
                if best_value_current_depth > best_value:
                    best_action = best_action_current_depth
                    best_value = best_value_current_depth

            # Dynamically increase depth if time allows
            depth += 1

        # exit()
        print(depth)
        return best_action if best_action is not None else random.choice(actions)