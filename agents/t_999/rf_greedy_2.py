# Import necessary modules and classes
from template import Agent
import random
from copy import deepcopy
from typing import List, Tuple
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from Azul.azul_utils import Action, TileGrab

# Constants for agent configuration
THINKTIME = 0.9  # Maximum allowed thinking time for the agent
NUM_PLAYERS = 2  # Number of players in the game

def GetScore(agent_id: int, game_state: AzulState) -> int:
    """
    Get the score for a specific agent in the given game state.

    Args:
        agent_id (int): The ID of the agent whose score is to be calculated.
        game_state (AzulState): The current state of the game.

    Returns:
        int: The calculated score for the agent.
    """
    agent_state = game_state.agents[agent_id]
    return ScoreState(agent_state)

def ScoreState(agent_state: AzulState.AgentState) -> int:
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
    wall_grid = deepcopy(agent_state.grid_state)  # Deep copy to avoid modifying the original grid
    current_score = agent_state.score  # Current score before calculations

    # Iterate over each pattern line
    for row in range(agent_state.GRID_SIZE):
        # Check if the pattern line is full
        if agent_state.lines_number[row] == row + 1:
            tile_color = agent_state.lines_tile[row]  # Color of the tile in the pattern line
            column = int(agent_state.grid_scheme[row][tile_color])  # Corresponding column in the wall grid

            # Place the tile on the wall grid
            wall_grid[row][column] = 1

            # Count adjacent tiles in all four directions
            adjacent_horizontal = (
                count_adjacent_tiles(wall_grid, row, column, 0, -1) +
                count_adjacent_tiles(wall_grid, row, column, 0, 1)
            )
            adjacent_vertical = (
                count_adjacent_tiles(wall_grid, row, column, -1, 0) +
                count_adjacent_tiles(wall_grid, row, column, 1, 0)
            )

            # Calculate points based on adjacency
            if adjacent_horizontal > 0 and adjacent_vertical > 0:
                points_gained += 1 + adjacent_horizontal + adjacent_vertical
            elif adjacent_horizontal > 0:
                points_gained += 1 + adjacent_horizontal
            elif adjacent_vertical > 0:
                points_gained += 1 + adjacent_vertical
            else:
                points_gained += 1  # Isolated tile

    # Calculate penalties for floor tiles
    floor_penalties = sum(tile * penalty for tile, penalty in zip(agent_state.floor, agent_state.FLOOR_SCORES))

    # Update the agent's score
    current_score += points_gained + floor_penalties

    return current_score

def count_adjacent_tiles(grid: List[List[int]], row: int, col: int,
                         row_step: int, col_step: int) -> int:
    """
    Count the number of adjacent tiles in a specific direction.

    Args:
        grid (List[List[int]]): The wall grid of the agent.
        row (int): The starting row index.
        col (int): The starting column index.
        row_step (int): The row increment (-1, 0, or 1).
        col_step (int): The column increment (-1, 0, or 1).

    Returns:
        int: The number of adjacent tiles in the specified direction.
    """
    count = 0
    row += row_step
    col += col_step

    # Traverse the grid in the specified direction
    while 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        if grid[row][col] == 1:
            count += 1
            row += row_step
            col += col_step
        else:
            break  # Stop if no tile is found

    return count

class myAgent(Agent):
    def __init__(self, _id: int):
        """
        Initialize the agent with a unique ID and game rules.

        Args:
            _id (int): The unique ID assigned to the agent.
        """
        self.id = _id  # Agent's unique identifier
        self.game_rule = GameRule(NUM_PLAYERS)  # Instance of the game rules

    def GetActions(self, state: AzulState) -> List[Tuple[Action, int, TileGrab]]:
        """
        Get a list of legal actions available to the agent in the current state.

        Args:
            state (AzulState): The current state of the game.

        Returns:
            List[Tuple[Action, int, TileGrab]]: A list of legal actions.
        """
        return self.game_rule.getLegalActions(state, self.id)

    def DoAction(self, state: AzulState, action: Tuple[Action, int, TileGrab]) -> bool:
        """
        Execute a given action on the game state and check if the game has ended.

        Args:
            state (AzulState): The current state of the game.
            action (Tuple[Action, int, TileGrab]): The action to execute.

        Returns:
            bool: True if the game has ended, False otherwise.
        """
        # Generate the next state after performing the action
        state = self.game_rule.generateSuccessor(state, action, self.id)

        # Check if there are no tiles remaining, indicating the end of the game
        goal_reached = not state.TilesRemaining()

        return goal_reached

    def SelectAction(self, actions: List[Tuple[Action, int, TileGrab]],
                     game_state: AzulState) -> Tuple[Action, int, TileGrab]:
        """
        Select the best action from the list of legal actions.

        The agent simulates the outcome of each action by calculating the
        potential score gain and also considers the opponent's potential
        moves to maximize its own advantage.

        Args:
            actions (List[Tuple[Action, int, TileGrab]]): List of legal actions.
            game_state (AzulState): The current state of the game.

        Returns:
            Tuple[Action, int, TileGrab]: The selected action.
        """
        best_score = float('-inf')
        best_action = random.choice(actions)  # Default action if all scores are equal

        for action in actions:
            # Simulate the agent's move
            simulated_state = deepcopy(game_state)
            simulated_state = self.game_rule.generateSuccessor(simulated_state, action, self.id)
            my_score = GetScore(self.id, simulated_state)
            net_score = my_score

            # Consider opponent's best possible response
            if len(actions) < 60:
                opponent_actions = self.game_rule.getLegalActions(simulated_state, 1 - self.id)
                max_opponent_score = float('-inf')

                for opp_action in opponent_actions:
                    # Simulate the opponent's move
                    opponent_state = deepcopy(simulated_state)
                    opponent_state = self.game_rule.generateSuccessor(opponent_state, opp_action, 1 - self.id)
                    opp_score = GetScore(1 - self.id, opponent_state)
                    max_opponent_score = max(max_opponent_score, opp_score)

                # Adjust net score based on opponent's best response
                net_score -= max_opponent_score

            # Update the best action based on net score
            if net_score > best_score:
                best_score = net_score
                best_action = action

        return best_action
