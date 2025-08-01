import math
import numpy as np
import torch

class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    Each node stores statistics for the edges leading out of it.
    """

    def __init__(self, parent=None):
        """
        Initializes an MCTSNode.
        :param parent: The parent node.
        """
        self.parent = parent
        self.children = {}  # A map from action to MCTSNode

        # N(s,a), W(s,a), Q(s,a), P(s,a) from the AlphaGo Zero paper
        self.visit_count = {}        # N: visit count for each action
        self.total_action_value = {} # W: total action-value for each action
        self.mean_action_value = {}  # Q: mean action-value for each action
        self.prior_prob = {}         # P: prior probability for each action

    def get_child(self, action):
        """
        Retrieves the child node corresponding to a given action.
        Matches the API used by mcts_rs.
        :param action: The action leading to the child node.
        :return: The child MCTSNode, or None if it doesn't exist.
        """
        return self.children.get(action)

    def is_leaf(self):
        """Checks if the node is a leaf node (i.e., has no children)."""
        return not self.children

    def select_action(self, c_puct):
        """
        Selects the action that maximizes the PUCT (Polynomial Upper Confidence for Trees) score.
        PUCT(s,a) = Q(s,a) + U(s,a)
        U(s, a) = c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a))

        :param c_puct: A constant determining the level of exploration.
        :return: The best action to take.
        """
        best_score = -float('inf')
        best_action = None

        # Total visits from this node is the sum of visits for all its actions
        total_visits_from_node = sum(self.visit_count.values())

        for action in self.children.keys():
            # Q(s,a) is the mean action value
            q_value = self.mean_action_value.get(action, 0.0)

            # U(s,a) is the exploration bonus
            u_value = (c_puct * self.prior_prob.get(action, 0) *
                       math.sqrt(total_visits_from_node) / (1 + self.visit_count.get(action, 0)))

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def expand(self, action_priors):
        """
        Expands the node by creating a child for each action with a prior probability.
        This assumes action_priors has already been filtered for legal moves.
        :param action_priors: A dictionary mapping each legal action to its prior probability.
        """
        for action, prob in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self)
                self.prior_prob[action] = prob
                self.visit_count[action] = 0
                self.total_action_value[action] = 0.0
                self.mean_action_value[action] = 0.0

    def update_stats_for_action(self, action, value):
        """
        Updates the statistics for a given action taken from this node.
        :param action: The action taken.
        :param value: The value (e.g., win/loss) from the simulation, from the perspective of the current player at this node.
        """
        if action not in self.visit_count: return
        self.visit_count[action] += 1
        self.total_action_value[action] += value
        self.mean_action_value[action] = self.total_action_value[action] / self.visit_count[action]


class MCTS:
    """
    An implementation of Batch Monte Carlo Tree Search as described in AlphaGo Zero.
    It collects leaf nodes from multiple simulations and evaluates them in a single batch.
    """

    def __init__(self, network, c_puct=1.0, dirichlet_alpha=0.03, epsilon=0.25):
        """
        :param network: A neural network that has a `predict(state_batch)` method,
                        which returns a tuple of (policy_batch, value_batch).
        :param c_puct: A constant determining the level of exploration.
        :param dirichlet_alpha: Alpha parameter for the Dirichlet noise.
        :param epsilon: Weight of the Dirichlet noise.
        """
        self.network = network
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon

    def run_simulations(self, root_node, root_state, num_simulations):
        """
        Run a batch of MCTS simulations starting from a root node.

        :param root_node: The root MCTSNode of the search tree.
        :param root_state: The game state corresponding to the root node.
        :param num_simulations: The number of simulations to perform.
        """
        # --- 1. Root Expansion and Noise ---
        if root_node.is_leaf():
            # For the very first search, expand the root and add noise.
            # This requires one initial network evaluation for the root.
            state_repr_batch = root_state.get_representation()
            policy_batch, value_batch = self.network.predict(state_repr_batch)
            self._expand_node(root_node, root_state, policy_batch[0])
            self._add_dirichlet_noise(root_node)

        leaves_to_evaluate = []
        for _ in range(num_simulations):
            # --- 2. Selection Phase ---
            # Traverse the tree from the root to a leaf node.
            path = []
            node = root_node
            state = root_state.clone()
            while not node.is_leaf():
                action = node.select_action(self.c_puct)
                if action is None: # Can happen if node has children but all have 0 prior
                    break
                path.append((node, action))
                node = node.children[action]
                state.apply_move(action)

            # --- 3. Terminal Node Check ---
            # `node` is now a leaf. Check if the game is over at this state.
            game_over, winner = state.is_game_over()
            if game_over:
                if winner == 0:
                    value = 0.0
                else:
                    # The value is from the perspective of the player about to move at the terminal state.
                    value = 1.0 if winner == state.get_current_player() else -1.0
                self._backup(path, value)
            else:
                # If not terminal, add to batch for network evaluation.
                leaves_to_evaluate.append({'path': path, 'leaf_node': node, 'state': state})
        # --- 4. Batch Expansion and Evaluation ---

        if leaves_to_evaluate:
            # Prepare batch for network
            state_representations = [item['state'].get_representation() for item in leaves_to_evaluate]
            batch_tensor = torch.cat(state_representations, dim=0)

            # Get network predictions for the entire batch
            policy_batch, value_batch = self.network.predict(batch_tensor)

            # Process each leaf in the batch
            for i, item in enumerate(leaves_to_evaluate):
                self._expand_node(item['leaf_node'], item['state'], policy_batch[i])
                # Backup the network's value prediction
                self._backup(item['path'], value_batch[i][0])

    def _add_dirichlet_noise(self, node):
        """Adds Dirichlet noise to a node's prior probabilities for exploration."""
        if not node.prior_prob: return
        actions = list(node.prior_prob.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, action in enumerate(actions):
            node.prior_prob[action] = (1 - self.epsilon) * node.prior_prob[action] + self.epsilon * noise[i]

    def _expand_node(self, node, state, policy_raw):
        """Expands a leaf node, creating children and setting their prior probabilities."""
        legal_actions = state.get_legal_moves()
        action_priors = {action: prob for action, prob in enumerate(policy_raw) if action in legal_actions}

        # Re-normalize probabilities over legal moves
        prob_sum = sum(action_priors.values())
        if prob_sum > 1e-6:
            for action in action_priors:
                action_priors[action] /= prob_sum
        else: # Fallback to uniform if network assigns ~0 to all legal moves
            num_legal = len(legal_actions)
            action_priors = {action: 1.0 / num_legal for action in legal_actions} if num_legal > 0 else {}

        node.expand(action_priors)

    def _backup(self, path, value):
        """Propagates the value back up the tree along the given path."""
        current_value = value
        for parent_node, action_taken in reversed(path):
            # The value perspective must be flipped for the parent.
            current_value = -current_value
            parent_node.update_stats_for_action(action_taken, current_value)

    def get_move_probs(self, root_node, temp=1.0):
        """
        Get the move probabilities from the root node after running simulations.
        pi(a|s0) is proportional to N(s0, a)^(1/tau)

        :param root_node: The root node of the search.
        :param temp: The temperature parameter `tau`.
        :return: A dictionary of {action: probability}.
        """
        visit_counts = root_node.visit_count

        if not visit_counts:
            return {}

        if temp == 0:
            # Deterministic play: choose the most visited move
            best_action = max(visit_counts, key=visit_counts.get)
            probs = {action: 0.0 for action in visit_counts}
            probs[best_action] = 1.0
            return probs

        # Apply temperature
        powered_counts = {action: count**(1.0 / temp) for action, count in visit_counts.items()}
        total_powered_count = sum(powered_counts.values())

        if total_powered_count < 1e-9:
            num_legal = len(visit_counts)
            return {action: 1.0 / num_legal for action in visit_counts} if num_legal > 0 else {}

        probs = {action: count / total_powered_count for action, count in powered_counts.items()}
        return probs
