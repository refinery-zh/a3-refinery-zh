from typing import List, Optional, Any
from agents.t_999.mcts_rain import MCTSNode, MCTS, myAgent
from agents.t_999.myTeam import *
import unittest
from template import GameState

from Azul.azul_utils import Tile
from Azul.azul_model import AzulState, AzulGameRule as GameRule
import numpy as np


num_agents = 2
class TestMCTSNode(unittest.TestCase):

    def setUp(self):
        self.azul_state = AzulState(num_agents)
        test_state = self.azul_state.AgentState(0)
        test_state.score = 0
        test_state.grid_scheme = test_state.grid_scheme.astype(np.int32)
        print(type(test_state.grid_scheme[0][Tile.BLUE]))
        test_state.grid_state[0][test_state.grid_scheme[0][Tile.BLUE]] = 1
        test_state.grid_state[0][test_state.grid_scheme[0][Tile.BLACK]] = 1
        test_state.grid_state[0][test_state.grid_scheme[0][Tile.YELLOW]] = 1
        test_state.grid_state[0][test_state.grid_scheme[0][Tile.RED]] = 1

        test_state.grid_state[4][test_state.grid_scheme[4][Tile.BLUE]] = 1
        test_state.grid_state[1][test_state.grid_scheme[1][Tile.BLACK]] = 1
        test_state.grid_state[3][test_state.grid_scheme[3][Tile.YELLOW]] = 1
        test_state.grid_state[2][test_state.grid_scheme[2][Tile.RED]] = 1

        test_state.floor[0] = 1

        test_state.lines_number[0] = 1
        test_state.lines_tile[0] = Tile.WHITE
        # print(test_state.__dict__)

        self.agent_state = test_state
        s = ScoreState(test_state)

        # def __init__(self, agent_id: int, state: AzulState, parent: Optional['MCTSNode'] = None) -> None:
        # self.state = state
        # self.parent = parent
        # self.children: List['MCTSNode'] = []
        # self.visits: int = 0
        # self.wins: float = 0
        # self.agent_id: int = agent_id

        self.node = MCTSNode(0,AzulState)
        

        
        self.s = s

    def test_score_state(self):
        self.assertEqual(ScoreState(self.agent_state), 9)

    def test_simulate(self):
        azul_state = self.azul_state
        node = self.node

        mct = MCTS()
        result = mct.simulate(node)
        self.assertEqual(result,[0,0])


        



if __name__ == '__main__':
    unittest.main()