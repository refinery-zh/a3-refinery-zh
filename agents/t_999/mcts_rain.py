from copy import deepcopy
from math import sqrt, log
from Azul.azul_model import AzulGameRule as GameRule
import random
import time

THINKTIME = 0.9
THINKTIME = 5
NUM_PLAYERS = 2

class MCTSNode:
    def __init__(self, agent_id, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.isExpanded = False
        self.wins = 0
        self.agent_id = agent_id

    def uct(self):
        if self.visits == 0:
            return float('inf')  # Avoid division by zero
        # UCB1 formula
        return (self.wins / self.visits) + sqrt(2 * log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        # A node is fully expanded if all possible actions have been taken as children
        return len(self.children) == len(GameRule(NUM_PLAYERS).getLegalActions(self.state, self.agent_id))


class MCTS:
    EXPANSION_THRESHOLD = 10
    def __init__(self):
        self.game_rule = GameRule(NUM_PLAYERS)

    def select(self, node):
        # Selection: keep traversing the child with the highest UCT value until a leaf node is reached
        while node.children:
            node = max(node.children, key=lambda n: n.uct())
        return node

    def expand(self, node):
        # Expansion: create a child for each unvisited action
        # print(f'Is fully expanded: {node.is_fully_expanded()}')
        # if not node.is_fully_expanded():
        if not node.isExpanded:
            node.isExpanded = True
            new_actions = self.game_rule.getLegalActions(node.state, node.agent_id)
            for action in new_actions:
                next_state = deepcopy(node.state) 
                # print(f'Action: {action}')
                new_state = self.game_rule.generateSuccessor(next_state, action, node.agent_id)
                # use another agent id
                new_node = MCTSNode((node.agent_id + 1) % NUM_PLAYERS, new_state, node)
                # print('Adding child')
                node.children.append(new_node)
                # print(f'Child: {new_node}')
            # print('Randomly selecting child')
            # return random.choice(node.children) if node.children else node
        return node

    def simulate(self, node):
        # Simulation: perform a random rollout from the current state to the end of the game
        game_state = deepcopy(node.state)
        current_agent_id = node.agent_id
        # print(f'Simulating from agent {current_agent_id}')
        while game_state.TilesRemaining():
            gs_copy = deepcopy(game_state)
            legal_actions = self.game_rule.getLegalActions(gs_copy, current_agent_id)
            # TODO: Implement a rollout policy
            action = random.choice(legal_actions)  # Random rollout
            game_state = self.game_rule.generateSuccessor(gs_copy, action, current_agent_id)
            current_agent_id = (current_agent_id + 1) % NUM_PLAYERS

        score_0 = game_state.agents[0].score
        score_1 = game_state.agents[1].score
        # print(f'Simulated scores: {score_0}, {score_1}')
        return [score_0, score_1]

    def backpropagate(self, node, result):
        # Backpropagation: update the visited nodes with the result
        while node:
            node.visits += 1
            # if node.agent_id == 0:  # Assuming agent 0 is the main agent
            #     node.wins += result[0] > result[1]  # Wins if agent 0 scores higher
            node.wins += result[0] > result[1]
            node = node.parent


class myAgent:
    def __init__(self, _id):
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule.
        self.mcts = MCTS()

    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        root = MCTSNode(self.id, rootstate)
        self.mcts.expand(root)
        # Perform MCTS iterations within the allowed think time
        cnt = 0
        while time.time() - start_time < THINKTIME:
            cnt += 1
            node = self.mcts.select(root)  # Selection step
            # print(f'Selected node: {node}')
            if node.visits >= MCTS.EXPANSION_THRESHOLD:
                node = self.mcts.expand(node)  # Expansion step
                node = self.mcts.select(node)
            # print(f'Node to simulate: {node}')
            result = self.mcts.simulate(node)  # Simulation step
            # print(result)
            self.mcts.backpropagate(node, result)  # Backpropagation step
        # Choose the best child node with the highest number of visits if timeout occurs
        if root.children:
            # print(root.children)
            best_child = max(root.children, key=lambda n: n.visits)
            best_action = self.game_rule.getLegalActions(root.state, self.id)[root.children.index(best_child)]
        else:
            # If no children were expanded, fallback to a random legal action
            print('timeout')
            best_action = random.choice(actions)
        print(cnt)
        return best_action