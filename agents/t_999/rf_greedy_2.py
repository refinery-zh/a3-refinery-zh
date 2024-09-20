from template import Agent
import time, random
import numpy as np
from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from copy import deepcopy
from typing import List, Tuple
from Azul.azul_utils import Action, TileGrab
from .scoring_utils import GetScore

# Constants
THINKTIME   = 0.9  # Maximum allowed thinking time
NUM_PLAYERS = 2    # Number of players in the game

class myAgent(Agent):
    def __init__(self, _id: int):
        self.id: int = _id  # Store the agent's player ID
        self.game_rule: GameRule = GameRule(NUM_PLAYERS)  # Instance of game rules to use helper functions

    def GetActions(self, state: AzulState) -> List[Tuple[Action, int, TileGrab]]:
        return self.game_rule.getLegalActions(state, self.id)
    
    # Carry out a given action on this state and return True if goal is reached.
    def DoAction(self, state: AzulState, action: Tuple[Action, int, TileGrab]) -> bool:
        state: AzulState = self.game_rule.generateSuccessor(state, action, self.id)
        goal_reached: bool = not state.TilesRemaining()
        return goal_reached
        
    def SelectAction(self, actions: List[Tuple[Action, int, TileGrab]], game_state: AzulState) -> Tuple[Action, int, TileGrab]:
        """
        Select the best action from the given list of legal actions while respecting the think time limit.

        Args:
            actions (List[Tuple[Action, int, TileGrab]]): List of legal actions.
            game_state (AzulState): Current state of the game.

        Returns: 
            Tuple[Action, int, TileGrab]: The selected action.
        """
        start_time = time.time()
        scores: List[float] = []
        for a in actions:
            # Check time limit
            if time.time() - start_time > THINKTIME:
                break  # Exit the loop if time limit is exceeded

            op_scores: List[float] = []
            gs: AzulState = deepcopy(game_state)
            gs = self.game_rule.generateSuccessor(gs, a, self.id)
            score: float = GetScore(self.id, gs)

            if len(actions) < 60:
                new_actions: List[Tuple[Action, int, TileGrab]] = self.game_rule.getLegalActions(gs, 1 - self.id)
                for na in new_actions:
                    # Check time limit
                    if time.time() - start_time > THINKTIME:
                        break  # Exit the inner loop if time limit is exceeded

                    ngs: AzulState = deepcopy(gs)
                    ngs = self.game_rule.generateSuccessor(ngs, na, 1 - self.id)
                    new_score: float = GetScore(1 - self.id, ngs)
                    op_scores.append(new_score)

                if op_scores:
                    ops: float = np.max(op_scores)
                    scores.append(score - ops)
                else:
                    # If no opponent scores were calculated, use own score
                    scores.append(score)
            else:
                scores.append(score)

        if scores:
            # Select the action with the maximum score
            return actions[np.argmax(scores)]
        else:
            # If time ran out before any actions were evaluated, choose a random action
            return random.choice(actions)
