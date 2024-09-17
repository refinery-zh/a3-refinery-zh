from copy import deepcopy
from math import sqrt, log
from Azul.azul_model import AzulGameRule as GameRule
import random
import time
import typing
from typing import Any, List, Optional
from Azul.azul_model import AzulState, AzulGameRule as GameRule


THINKTIME = 0.9
NUM_PLAYERS = 2

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

    def uct(self) -> float:
        """
        Calculate the Upper Confidence Bound 1 (UCB1) value for this node.

        Returns:
            float: The UCB1 value.

        Example:
            >>> node = MCTSNode(agent_id=0, state=AzulState)
            >>> node.visits = 10
            >>> node.wins = 7
            >>> node.parent = MCTSNode(agent_id=1, state=AzulState)
            >>> node.parent.visits = 100
            >>> print(f"{node.uct():.4f}")
            0.8839  # This value may vary
        """
        if self.visits == 0:
            return float('inf')  # Avoid division by zero
        # UCB1 formula
        return (self.wins / self.visits) + sqrt(2 * log(self.parent.visits) / self.visits)

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
    """
    Implements the Monte Carlo Tree Search algorithm.

    Attributes:
        game_rule (GameRule): An instance of the game rules.

    Example:
        >>> mcts = MCTS()
        >>> root_node = MCTSNode(agent_id=0, state=AzulState)
        >>> best_child = mcts.select(root_node)
    """

    def __init__(self) -> None:
        self.game_rule = GameRule(NUM_PLAYERS)

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
            node = max(node.children, key=lambda n: n.uct())
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
        print(f'Is fully expanded: {node.is_fully_expanded()}')
        if not node.is_fully_expanded():
            new_actions = self.game_rule.getLegalActions(node.state, node.agent_id)
            for action in new_actions:
                next_state = deepcopy(node.state) 
                print(f'Action: {action}')
                new_state = self.game_rule.generateSuccessor(next_state, action, node.agent_id)
                # use another agent id
                new_node = MCTSNode((node.agent_id + 1) % NUM_PLAYERS, new_state, node)
                print('Adding child')
                node.children.append(new_node)
                print(f'Child: {new_node}')
            print('Randomly selecting child')
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
        game_state = deepcopy(node.state)
        current_agent_id = node.agent_id
        print(f'Simulating from agent {current_agent_id}')
        while game_state.TilesRemaining():
            gs_copy = deepcopy(game_state)
            legal_actions = self.game_rule.getLegalActions(gs_copy, current_agent_id)
            # TODO: Implement a rollout policy
            action = random.choice(legal_actions)  # Random rollout
            game_state = self.game_rule.generateSuccessor(gs_copy, action, current_agent_id)
            current_agent_id = (current_agent_id + 1) % NUM_PLAYERS

        score_0 = game_state.agents[0].score
        score_1 = game_state.agents[1].score
        print(f'Simulated scores: {score_0}, {score_1}')
        return [score_0, score_1]

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
            if node.agent_id == 0:  # Assuming agent 0 is the main agent
                node.wins += result[0] > result[1]  # Wins if agent 0 scores higher
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
        while time.time() - start_time < THINKTIME:
            node = self.mcts.select(root)  # Selection step
            print(f'Selected node: {node}')
            node = self.mcts.expand(node)  # Expansion step
            print(f'Node to simulate: {node}')
            result = self.mcts.simulate(node)  # Simulation step
            print(result)
            self.mcts.backpropagate(node, result)  # Backpropagation step
        # Choose the best child node with the highest number of visits if timeout occurs
        if root.children:
            print(root.children)
            best_child = max(root.children, key=lambda n: n.visits)
            best_action = self.game_rule.getLegalActions(root.state, self.id)[root.children.index(best_child)]
        else:
            # If no children were expanded, fallback to a random legal action
            print('timeout')
            best_action = random.choice(actions)

        return best_action
    

    __all__ = ['MCTSNode', 'MCTS','myAgent']