from template import Agent
import time, random
import numpy as np
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from copy import deepcopy
from collections import deque
from typing import List, Dict, Tuple
from Azul.azul_utils import Action, TileGrab
from .scoring_utils import GetScore, ScoreState, count_adjacent_tiles
#try3


THINKTIME   = 0.9
NUM_PLAYERS = 2


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
