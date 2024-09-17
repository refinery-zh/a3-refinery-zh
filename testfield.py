from Azul.azul_utils import Tile
from agents.t_999.myTeam import *
from Azul.azul_model import AzulState
import numpy as np
if __name__ == "__main__":
    # test_state = AzulState.AgentState(0)
    # test_state.score = 0
    # test_state.grid_scheme = test_state.grid_scheme.astype(np.int32)
    # print(type(test_state.grid_scheme[0][Tile.BLUE]))
    # test_state.grid_state[0][test_state.grid_scheme[0][Tile.BLUE]] = 1
    # test_state.grid_state[0][test_state.grid_scheme[0][Tile.BLACK]] = 1
    # test_state.grid_state[0][test_state.grid_scheme[0][Tile.YELLOW]] = 1
    # test_state.grid_state[0][test_state.grid_scheme[0][Tile.RED]] = 1

    # test_state.grid_state[4][test_state.grid_scheme[4][Tile.BLUE]] = 1
    # test_state.grid_state[1][test_state.grid_scheme[1][Tile.BLACK]] = 1
    # test_state.grid_state[3][test_state.grid_scheme[3][Tile.YELLOW]] = 1
    # test_state.grid_state[2][test_state.grid_scheme[2][Tile.RED]] = 1
    # test_state.grid_state[0][test_state.grid_scheme[0][Tile.YELLOW]] = 1
    # test_state.grid_state[0][test_state.grid_scheme[0][Tile.BLACK]] = 1
    # test_state.grid_state[1][test_state.grid_scheme[1][Tile.WHITE]] = 1
    # test_state.grid_state[2][test_state.grid_scheme[2][Tile.BLUE]] = 1
    # test_state.grid_state[3][test_state.grid_scheme[3][Tile.RED]] = 1
    # test_state.grid_state[3][test_state.grid_scheme[3][Tile.BLACK]] = 1
    # test_state.grid_state[3][test_state.grid_scheme[3][Tile.WHITE]] = 1
    # test_state.grid_state[3][test_state.grid_scheme[3][Tile.BLUE]] = 1
    # test_state.floor[0] = 1


    # test_state.lines_number[0] = 1
    # test_state.lines_tile[0] = Tile.WHITE
    # print(test_state.__dict__)

    
    
    agent_state_0 = AzulState.AgentState(0)
    agent_state_0.score = 0
    agent_state_0.grid_scheme = agent_state_0.grid_scheme.astype(np.int32)
    agent_state_0.grid_state[0][agent_state_0.grid_scheme[0][Tile.YELLOW]] = 0
    agent_state_0.grid_state[0][agent_state_0.grid_scheme[0][Tile.BLACK]] = 0
    agent_state_0.grid_state[0][agent_state_0.grid_scheme[0][Tile.RED]] = 0
    agent_state_0.grid_state[0][agent_state_0.grid_scheme[0][Tile.WHITE]] = 0
    agent_state_0.grid_state[0][agent_state_0.grid_scheme[0][Tile.BLUE]] = 0

    agent_state_0.grid_state[1][agent_state_0.grid_scheme[1][Tile.WHITE]] = 0
    agent_state_0.grid_state[1][agent_state_0.grid_scheme[1][Tile.BLACK]] = 0
    agent_state_0.grid_state[1][agent_state_0.grid_scheme[1][Tile.RED]] = 0
    agent_state_0.grid_state[1][agent_state_0.grid_scheme[1][Tile.BLUE]] = 0
    agent_state_0.grid_state[1][agent_state_0.grid_scheme[1][Tile.YELLOW]] = 0

    
    agent_state_0.grid_state[2][agent_state_0.grid_scheme[2][Tile.BLUE]] = 0
    agent_state_0.grid_state[2][agent_state_0.grid_scheme[2][Tile.YELLOW]] = 0
    agent_state_0.grid_state[2][agent_state_0.grid_scheme[2][Tile.BLACK]] = 0
    agent_state_0.grid_state[2][agent_state_0.grid_scheme[2][Tile.RED]] = 0
    agent_state_0.grid_state[2][agent_state_0.grid_scheme[2][Tile.WHITE]] = 0

    agent_state_0.grid_state[3][agent_state_0.grid_scheme[3][Tile.RED]] = 0
    agent_state_0.grid_state[3][agent_state_0.grid_scheme[3][Tile.BLACK]] = 0
    agent_state_0.grid_state[3][agent_state_0.grid_scheme[3][Tile.WHITE]] = 0
    agent_state_0.grid_state[3][agent_state_0.grid_scheme[3][Tile.BLUE]] = 0
    agent_state_0.grid_state[3][agent_state_0.grid_scheme[3][Tile.YELLOW]] = 0

    agent_state_0.grid_state[4][agent_state_0.grid_scheme[4][Tile.YELLOW]] = 0
    agent_state_0.grid_state[4][agent_state_0.grid_scheme[4][Tile.BLACK]] = 0
    agent_state_0.grid_state[4][agent_state_0.grid_scheme[4][Tile.RED]] = 0
    agent_state_0.grid_state[4][agent_state_0.grid_scheme[4][Tile.WHITE]] = 0
    agent_state_0.grid_state[4][agent_state_0.grid_scheme[4][Tile.BLUE]] = 0
    agent_state_0.floor[0] = 0

    agent_state_10 = AzulState.AgentState(10)
    agent_state_10.score = 10
    agent_state_10.grid_state[0][agent_state_10.grid_scheme[0][Tile.YELLOW]] = 1
    agent_state_10.grid_state[0][agent_state_10.grid_scheme[0][Tile.BLACK]] = 1
    agent_state_10.grid_state[1][agent_state_10.grid_scheme[1][Tile.WHITE]] = 1
    agent_state_10.grid_state[2][agent_state_10.grid_scheme[2][Tile.BLUE]] = 1
    agent_state_10.grid_state[3][agent_state_10.grid_scheme[3][Tile.RED]] = 1
    agent_state_10.grid_state[3][agent_state_10.grid_scheme[3][Tile.BLACK]] = 1
    agent_state_10.grid_state[3][agent_state_10.grid_scheme[3][Tile.WHITE]] = 1
    agent_state_10.grid_state[3][agent_state_10.grid_scheme[3][Tile.BLUE]] = 1
    agent_state_10.floor[0] = 1





    s = ScoreState(agent_state_0)
    print(s)
    exit()





