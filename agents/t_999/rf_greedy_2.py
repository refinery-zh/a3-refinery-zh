from template import Agent
import time, random
import numpy as np
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from copy import deepcopy
from collections import deque

THINKTIME   = 0.9
NUM_PLAYERS = 2

def GetScore(id, game_state:AzulState):
    my_gs = game_state.agents[id]
    return ScoreState(my_gs)

def ScoreState(agent_state: AzulState.AgentState):
    """
    Calculate the score for a given agent state in Azul.

    This function evaluates the current state of an agent's board,
    calculating points for completed rows, columns, and sets,
    as well as applying penalties for floor tiles.

    Args:
        agent_state (AzulState.AgentState): The current state of the agent.

    Returns:
        int: The calculated score for the agent.
    """
    points_gained = 0
    wall_grid = deepcopy(agent_state.grid_state)
    current_score = agent_state.score

    # 1. Move tiles from pattern lines to the wall grid and calculate points
    for row in range(agent_state.GRID_SIZE):
        # Check if the pattern line is full
        if agent_state.lines_number[row] == row + 1:
            tile_color = agent_state.lines_tile[row]
            column = int(agent_state.grid_scheme[row][tile_color])

            # Place tile on the wall grid
            wall_grid[row][column] = 1

            # Count adjacent tiles in all directions
            adjacent_up = count_adjacent_tiles(wall_grid, row, column, -1, 0)
            adjacent_down = count_adjacent_tiles(wall_grid, row, column, 1, 0)
            adjacent_left = count_adjacent_tiles(wall_grid, row, column, 0, -1)
            adjacent_right = count_adjacent_tiles(wall_grid, row, column, 0, 1)

            # Calculate points for vertical line
            if adjacent_up > 0 or adjacent_down > 0:
                points_gained += (1 + adjacent_up + adjacent_down)

            # Calculate points for horizontal line
            if adjacent_left > 0 or adjacent_right > 0:
                points_gained += (1 + adjacent_left + adjacent_right)

            # Award 1 point if tile is isolated
            if adjacent_up == 0 and adjacent_down == 0 and adjacent_left == 0 and adjacent_right == 0:
                points_gained += 1

    # Calculate penalties for floor tiles
    floor_penalties = sum(tile * penalty for tile, penalty in zip(agent_state.floor, agent_state.FLOOR_SCORES))
    
    # Update score
    score_change = points_gained + floor_penalties
    current_score += score_change
    
    return current_score

def count_adjacent_tiles(grid, row, col, row_step, col_step):
    """
    Count the number of adjacent tiles in a given direction.

    Args:
        grid (list): The wall grid.
        row (int): Starting row.
        col (int): Starting column.
        row_step (int): Step for row direction (-1, 0, or 1).
        col_step (int): Step for column direction (-1, 0, or 1).

    Returns:
        int: Number of adjacent tiles in the specified direction.
    """
    count = 0
    row += row_step
    col += col_step
    while 0 <= row < len(grid) and 0 <= col < len(grid):
        if grid[row][col] == 1:
            count += 1
        else:
            break
        row += row_step
        col += col_step
    return count

class myAgent(Agent):
    def __init__(self, _id: int):
        self.id = _id
        self.game_rule = GameRule(NUM_PLAYERS)

    def GetActions(self, state: AzulState) -> List[str]:
        return self.game_rule.getLegalActions(state, self.id)
    
    def DoAction(self, state: AzulState, action: str) -> bool:
        state = self.game_rule.generateSuccessor(state, action, self.id)
        return not state.TilesRemaining()
        
    def SelectAction(self, actions: List[str], game_state: AzulState) -> str:
        def evaluate_action(action: str) -> float:
            gs = self.game_rule.generateSuccessor(deepcopy(game_state), action, self.id)
            score = GetScore(self.id, gs)
            
            if len(actions) < 60:
                opponent_actions = self.game_rule.getLegalActions(gs, 1-self.id)
                opponent_scores = [GetScore(1-self.id, self.game_rule.generateSuccessor(deepcopy(gs), a, 1-self.id)) for a in opponent_actions]
                return score - max(opponent_scores, default=0)
            
            return score

        return max(actions, key=evaluate_action)

