from copy import deepcopy
from math import sqrt, log
from Azul.azul_model import AzulGameRule as GameRule
import random
import time
import numpy as np
import typing
from typing import Any, List, Optional
from Azul.azul_model import AzulState, AzulGameRule as GameRule


THINKTIME = 0.9
NUM_PLAYERS = 2

def ScoreState(state):
    score_inc = 0
    grid_state = deepcopy(state.grid_state)

    # 1. Action tiles across from pattern lines to the wall grid
    for i in range(state.GRID_SIZE):
        # Is the pattern line full? If not it persists in its current
        # state into the next round.
        ori_score = score_inc
        if state.lines_number[i] == i+1:
            tc = state.lines_tile[i]
            col = int(state.grid_scheme[i][tc])
            # print(tc, col)
            # Tile will be placed at position (i,col) in grid
            grid_state[i][col] = 1

            # count the number of tiles in a continguous line
            # above, below, to the left and right of the placed tile.
            above = 0
            for j in range(col-1, -1, -1):
                val = grid_state[i][j]
                above += val
                if val == 0:
                    break
            below = 0
            for j in range(col+1,state.GRID_SIZE,1):
                val = grid_state[i][j]
                below +=  val
                if val == 0:
                    break
            left = 0
            for j in range(i-1, -1, -1):
                val = grid_state[j][col]
                left += val
                if val == 0:
                    break
            right = 0
            for j in range(i+1, state.GRID_SIZE, 1):
                val = grid_state[j][col]
                right += val
                if val == 0:
                    break

            # If the tile sits in a contiguous vertical line of 
            # tiles in the grid, it is worth 1*the number of tiles
            # in this line (including itstate).
            if above > 0 or below > 0:
                score_inc += (1 + above + below)

            # In addition to the vertical score, the tile is worth
            # an additional H points where H is the length of the 
            # horizontal contiguous line in which it sits.
            if left > 0 or right > 0:
                score_inc += (1 + left + right)

            # If the tile is not next to any already placed tiles
            # on the grid, it is worth 1 point.                
            if above == 0 and below == 0 and left == 0 and right == 0:
                score_inc += 1

            # print(score_inc - ori_score)

    # Score penalties for tiles in floor line
    penalties = 0
    for i in range(len(state.floor)):
        penalties += state.floor[i]*state.FLOOR_SCORES[i]
    # print(f'From ScoreState: {score_inc}, {penalties}')
    # Agents cannot be assigned a negative score in any round.
    score_change = score_inc + penalties
    return score_change

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search.

    Attributes:
        state (Any): The game state at this node.
        parent (Optional[MCTSNode]): The parent node, or None if this is the root.
        children (List[MCTSNode]): List of child nodes.
        visits (int): Number of times this node has been visited.
        wins (float): Number of wins accumulated through this node.
        agent_id (int): The ID of the agent at this node.

    Example:
        >>> game_state = GameState()  # Assuming a GameState class exists
        >>> root_node = MCTSNode(agent_id=0, state=game_state)
        >>> child_node = MCTSNode(agent_id=1, state=game_state, parent=root_node)
        >>> root_node.children.append(child_node)

        new_node = MCTSNode((node.agent_id + 1) % NUM_PLAYERS, new_state, node)
                
    """

    def __init__(self, agent_id: int, state: AzulState, parent: Optional['MCTSNode'] = None) -> None:
        self.state = state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.wins: float = 0
        self.agent_id: int = agent_id
        simulated_time_0 = 0
        simulated_time_1 = 0

    def uct(self, exploration_param) -> float:

        if self.visits == 0:
            return float('inf')  # Avoid division by zero
        # UCB1 formula
        return (self.wins / self.visits) + exploration_param * sqrt(log(self.parent.visits) / self.visits)

    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions from this node have been explored.

        Returns:
            bool: True if the node is fully expanded, False otherwise.

        Example:
            >>> node = MCTSNode(agent_id=0, state=AzulState)
            >>> print(node.is_fully_expanded())
            False  # Assuming no children have been added yet
        """
        # A node is fully expanded if all possible actions have been taken as children
        return len(self.children) == len(GameRule(NUM_PLAYERS).getLegalActions(self.state, self.agent_id))


class MCTS:
    def __init__(self, exploration_param=1.41):
        self.game_rule = GameRule(NUM_PLAYERS)
        self.exploration_param = exploration_param

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to expand using the UCT algorithm.

        Args:
            node (MCTSNode): The starting node for selection.

        Returns:
            MCTSNode: The selected node for expansion.

        Example:
            >>> mcts = MCTS()
            >>> root = MCTSNode(agent_id=0, state=GameState())
            >>> selected_node = mcts.select(root)
        """
        # Selection: keep traversing the child with the highest UCT value until a leaf node is reached
        while node.children:
            node = max(node.children, key=lambda n: n.uct(self.exploration_param))
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand the given node by adding a child node for an unexplored action.

        Args:
            node (MCTSNode): The node to expand.

        Returns:
            MCTSNode: Either a new child node or the input node if fully expanded.

        Example:
            >>> mcts = MCTS()
            >>> node = MCTSNode(agent_id=0, state=AzulState)
            >>> expanded_node = mcts.expand(node)
        """
        # Expansion: create a child for each unvisited action
        # print(f'Is fully expanded: {node.is_fully_expanded()}')
        if not node.is_fully_expanded():
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
            return random.choice(node.children) if node.children else node
        return node

    def simulate(self, node: MCTSNode) -> List[int]:
        """
        Simulate a game from the given node to completion using random moves.

        Args:
            node (MCTSNode): The starting node for the simulation.

        Returns:
            List[int]: A list of final scores for each player.

        Example:
            >>> mcts = MCTS()
            >>> node = MCTSNode(agent_id=0, state=AzulState)
            >>> final_scores = mcts.simulate(node)
            >>> print(final_scores)
            [42, 37]  # Example scores
        """
        # Simulation: perform a random rollout from the current state to the end of the game
        ori_score_0 = ScoreState(node.state.agents[0])
        ori_score_1 = ScoreState(node.state.agents[1])
        game_state = deepcopy(node.state)
        current_agent_id = node.agent_id
        if current_agent_id == 0:
            self.simulated_time_0 += 1
        else:
            self.simulated_time_1 += 1
        # print(f'Simulating from agent {current_agent_id}')
        # TODO: something wrong with game_state?
        while game_state.TilesRemaining():
            gs_copy = deepcopy(game_state)
            legal_actions = self.game_rule.getLegalActions(gs_copy, current_agent_id)
            # TODO: Implement a rollout policy
            original_score = gs_copy.agents[current_agent_id].score
            scores = []
            for a in legal_actions:
                new_gs = deepcopy(gs_copy)
                self.game_rule.generateSuccessor(new_gs, a, current_agent_id)
                new_score = ScoreState(new_gs.agents[current_agent_id])
                scores.append(new_score-original_score)
            action = legal_actions[np.argmax(scores)]
            # action = random.choice(legal_actions)
            game_state = self.game_rule.generateSuccessor(gs_copy, action, current_agent_id)
            current_agent_id = (current_agent_id + 1) % NUM_PLAYERS

        # TODO: need to deduct original score
        # score_0 = game_state.agents[0].score
        # score_1 = game_state.agents[1].score
        # TODO: need to consider future rewards
        # for agent_id in range(NUM_PLAYERS):
        #     print(f'Agent {agent_id}:')
        #     print(game_state.agents[agent_id].lines_number)
        #     print(game_state.agents[agent_id].lines_tile)
        #     print(game_state.agents[agent_id].grid_state)
        #     print(game_state.agents[agent_id].grid_scheme)
        #     print(game_state.agents[agent_id].floor)
        new_scorestate = [ScoreState(game_state.agents[0]) - ori_score_0, ScoreState(game_state.agents[1]) - ori_score_1]
        # print(f'Result: {new_scorestate}')
        # print('-'*15)
        return new_scorestate

    def backpropagate(self, node: MCTSNode, result: List[int]) -> None:
        """
        Update the statistics of the nodes in the path from the given node to the root.

        Args:
            node (MCTSNode): The leaf node to start backpropagation from.
            result (List[int]): The simulation result (scores) to backpropagate.

        Example:
            >>> mcts = MCTS()
            >>> leaf_node = MCTSNode(agent_id=1, state=AzulState)
            >>> mcts.backpropagate(leaf_node, [42, 37])
        """
        # Backpropagation: update the visited nodes with the result
        while node:
            node.visits += 1
            if (node.agent_id == 0 and result[0] > result[1]) or (node.agent_id == 1 and result[1] > result[0]):
                node.wins += 1  # Current agent wins
            elif result[0] == result[1]:
                node.wins += 0.5  # Draw
            node = node.parent

class myAgent:
    def __init__(self, _id):
        self.id = _id  # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS)  # Agent stores an instance of GameRule.
        self.mcts = MCTS()

    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        root = MCTSNode(self.id, rootstate)
        # Perform MCTS iterations within the allowed think time
        self.mcts.simulated_time_0 = 0
        self.mcts.simulated_time_1 = 0
        while time.time() - start_time < THINKTIME:
            node = self.mcts.select(root)  # Selection step
            # print(f'Selected node: {node}')
            node = self.mcts.expand(node)  # Expansion step
            # print(f'Node to simulate: {node}')
            result = self.mcts.simulate(node)  # Simulation step
            # print(result)
            self.mcts.backpropagate(node, result)  # Backpropagation step
        print(f'Simulated time 0: {self.mcts.simulated_time_0}')
        print(f'Simulated time 1: {self.mcts.simulated_time_1}')
        # Choose the best child node with the highest number of visits if timeout occurs
        if root.children:
            # for child in root.children:
                # print(f'Child: {child}')
                # print(f'Visits: {child.visits}')
                # print(f'Wins: {child.wins}')
                # print(f'UCT: {child.uct(1.41)}')
                # print(f'Agent took action: {self.game_rule.getLegalActions(root.state, self.id)[root.children.index(child)]}')
                # print(f'Tiles on the floor: {root.state.floor_line}')
            
            # get the score of this action only
            # avoid getting action leading to negative score
            indexed_children = [(child, i) for i, child in enumerate(root.children)]
            filtered_children = [(child, i) for child, i in indexed_children if ScoreState(child.state.agents[self.id]) >= root.state.agents[self.id].score]
            if filtered_children:
                print('Selected from filtered children')
                best_child, best_child_index = max(filtered_children, key=lambda n: n[0].uct(self.mcts.exploration_param))
            else:
                best_child, best_child_index = max(indexed_children, key=lambda n: n[0].uct(self.mcts.exploration_param))
            best_child_score = ScoreState(best_child.state.agents[self.id]) - root.state.agents[self.id].score - (ScoreState(root.state.agents[self.id]) - root.state.agents[self.id].score)
            print(f'Action score: {ScoreState(best_child.state.agents[self.id])}, Original score: {root.state.agents[self.id].score}, Existing tiles score: {ScoreState(root.state.agents[self.id]) - root.state.agents[self.id].score}')
            print(f'Best child score: {best_child_score}')
            best_action = self.game_rule.getLegalActions(root.state, self.id)[best_child_index]
            print(f'Best action: {best_action}')
        else:
            # If no children were expanded, fallback to a random legal action
            print('timeout')
            best_action = random.choice(actions)

        return best_action
    

    __all__ = ['MCTSNode', 'MCTS','myAgent']