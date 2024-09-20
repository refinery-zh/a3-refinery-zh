# INFORMATION ------------------------------------------------------------------------------------------------------- #
# Author:  Steven Spratley
# Date:    04/01/2021
# Purpose: Implements an example breadth-first search agent for the COMP90054 competitive game environment.
# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import time
import random
from typing import List, Tuple, Optional
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
import Azul.azul_utils as utils
from copy import deepcopy
import numpy as np
from .scoring_utils import GetScore

THINKTIME = 0.9
NUM_PLAYERS = 2

# FUNCTIONS ----------------------------------------------------------------------------------------------------------#
# Defines this agent.
class myAgent:
    def __init__(self, _id: int):
        """
        Initialize the agent with a given ID.

        Args:
            _id (int): The ID of the agent.
        """
        self.id: int = _id
        self.game_rule: GameRule = GameRule(NUM_PLAYERS)
        self.time_limit: float = 0.85

    def dfs_explore(self, actions: List[str], game_state: AzulState, depth: int, id: int, alpha: float, beta: float, end_time: float) -> Tuple[Optional[str], int]:
        """
        Perform a depth-first search with alpha-beta pruning.

        Args:
            actions (List[str]): List of possible actions.
            game_state (AzulState): The current state of the game.
            depth (int): The current depth of the search.
            id (int): The ID of the current agent.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.
            end_time (float): The time limit for the search.

        Returns:
            Tuple[Optional[str], int]: The best action and its evaluation score.
        """
        if depth == 0 or time.time() > end_time:
            return None, GetScore(self.id, game_state)
        
        best_action: Optional[str] = None
        if self.id == id:
            max_eval = float('-inf')
            for a in actions:
                gs = deepcopy(game_state)
                gs = self.game_rule.generateSuccessor(gs, a, id)
                eval = GetScore(self.id, gs) if a == "ENDROUND" else self.dfs_explore(self.game_rule.getLegalActions(gs, 1 - id), gs, depth - 1, 1 - id, alpha, beta, end_time)[1]
                if eval > max_eval:
                    max_eval = eval
                    best_action = a
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_action, max_eval
        else:
            min_eval = float('inf')
            for a in actions:
                gs = deepcopy(game_state)
                gs = self.game_rule.generateSuccessor(gs, a, id)
                eval = GetScore(self.id, gs) if a == "ENDROUND" else self.dfs_explore(self.game_rule.getLegalActions(gs, 1 - id), gs, depth - 1, 1 - id, alpha, beta, end_time)[1]
                if eval < min_eval:
                    min_eval = eval
                    best_action = a
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_action, min_eval

    def SelectAction(self, actions: List[str], game_state: AzulState) -> str:
        """
        Select the best action for the agent to take.

        Args:
            actions (List[str]): List of possible actions.
            game_state (AzulState): The current state of the game.

        Returns:
            str: The selected action.
        """
        start_time = time.time()
        end_time = start_time + self.time_limit
        depth_index = [2000,55,15,10,8,5,5,5,5,5,-1]
        len_actions = len(actions)
        for i in range(len(depth_index)):
            if len_actions > depth_index[i]:
                depth = i
                break
        best_action: Optional[str] = None

        action, _ = self.dfs_explore(actions, game_state, depth, self.id, float('-inf'), float('inf'), end_time)
        if action is not None:
            best_action = action

        return best_action if best_action is not None else random.choice(actions)

# END FILE -----------------------------------------------------------------------------------------------------------#