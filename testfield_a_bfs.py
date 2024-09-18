from Azul.azul_model import AzulGameRule as GameRule
from Azul.azul_model import AzulState
from agents.t_999.example_bfs import myAgent
from agents.generic.random import myAgent as dummy


if __name__ == "__main__":
    state = AzulState(2)
    game_rule = GameRule(2)
    # print(state.first_agent)
    # exit()
    first_agent = myAgent(state.first_agent)
    second_agent = dummy(1 - state.first_agent)
    state = game_rule.generateSuccessor(state, "STARTROUND", first_agent.id)
    # print(state.agents[state.first_agent].agent_trace.actions)
    while True:
        if not state.TilesRemaining():
            break

        actions = game_rule.getLegalActions(state, first_agent.id)
        action = first_agent.SelectAction(actions, state)
        print(action)
        state = game_rule.generateSuccessor(state, action, first_agent.id)
        if not state.TilesRemaining():
            break
        actions = game_rule.getLegalActions(state, 1 - first_agent.id)
        action = second_agent.SelectAction(actions, state)
        print(action)
        state = game_rule.generateSuccessor(state, action, 1 - first_agent.id)
