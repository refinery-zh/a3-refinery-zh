from template import Agent
import time, random
import numpy as np
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from copy import deepcopy
from collections import deque
from typing import List, Dict, Tuple
from Azul.azul_utils import Action, TileGrab

#try3


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

def count_adjacent_tiles(grid: List[List[int]], row: int, col: int, row_step: int, col_step: int) -> int:
    """
    Count the number of adjacent tiles in a given direction.

    Args:
        grid (List[List[int]]): The wall grid.
        row (int): Starting row.
        col (int): Starting column.
        row_step (int): Step for row direction (-1, 0, or 1).
        col_step (int): Step for column direction (-1, 0, or 1).

    Returns:
        int: Number of adjacent tiles in the specified direction.
    """
    count: int = 0
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
        self.id: int = _id  # Agent needs to remember its own id.
        self.game_rule: GameRule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule, from which to obtain functions.

    def GetActions(self, state: AzulState) -> List[Tuple[Action, int, TileGrab]]:
        return self.game_rule.getLegalActions(state, self.id)
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state: AzulState, action: Tuple[Action, int, TileGrab]) -> bool:
        state: AzulState = self.game_rule.generateSuccessor(state, action, self.id)
        
        # goal_reached = False #TODO: Students, how should agent check whether it reached goal or not
        goal_reached: bool = not state.TilesRemaining()

        return goal_reached
        
    def SelectAction(self, actions: List[Tuple[Action, int, TileGrab]], game_state: AzulState) -> Tuple[Action, int, TileGrab]:
        """
        Select the best action from the given list of legal actions.

        Args:
            actions (List[Tuple[Action, int, TileGrab]]): List of legal actions.
                Example: [(<Action.TAKE_FROM_FACTORY: 1>, 0, <Azul.azul_utils.TileGrab object>)]
            game_state (AzulState): Current state of the game.

        Returns: 
            Tuple[Action, int, TileGrab]: The selected action.
        """
        scores: List[float] = []
        for a in actions:
            op_scores: List[float] = []
            gs: AzulState = deepcopy(game_state)
            gs = self.game_rule.generateSuccessor(gs, a, self.id)
            score: float = GetScore(self.id, gs)
            if len(actions) < 60:
                new_actions: List[Tuple[Action, int, TileGrab]] = self.game_rule.getLegalActions(gs, 1-self.id)
                for na in new_actions:
                    ngs: AzulState = deepcopy(gs)
                    ngs = self.game_rule.generateSuccessor(ngs, na, 1-self.id)
                    new_score: float = GetScore(1-self.id, ngs)
                    op_scores.append(new_score)
                ops: float = np.max(op_scores)
                scores.append(score - ops)
            else:
                scores.append(score)
        return actions[np.argmax(scores)]
        # return random.choice(actions)
