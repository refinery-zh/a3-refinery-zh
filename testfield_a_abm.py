from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from agents.t_999.rf_minimax_e_abm import myAgent


if __name__ == "__main__":
    state = AzulState(2)
    game_rule = GameRule(2)
    # print(state.first_agent)
    # exit()
    first_agent = myAgent(state.first_agent)
    second_agent = myAgent(1-state.first_agent)
    actions = game_rule.getLegalActions(state, first_agent.id)
    print(actions)
    first_agent.SelectAction(actions, state)