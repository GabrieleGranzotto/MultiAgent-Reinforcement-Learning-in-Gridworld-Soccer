from __future__ import annotations
import numpy as np

class BeliefLearner:
    """
    A reinforcement learning agent that learns to play football using Q-learning and belief tracking.
    """
    def __init__(self, gamma: float = 1.0, rng: np.random.Generator | None = None):
        """
        Initialize the BeliefLearner agent.

        Args:
            gamma (float): The discount factor for future rewards.
            rng (np.random.Generator | None): The random number generator to use.
        """
        self.q_table = {}
        self.belief_traker = {}
        self.actions = {"0": 0, "N": 1, "S": 2, "E": 3, "W": 4}
        self.epsilon = 0.2 # Exploration rate
        self.alpha = 1.0 # Learning rate
        self.gamma = gamma # Discount factor
        self.rng = rng if rng is not None else np.random.default_rng()

    def _get_or_create_state(self, state):
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros((5, 5))  # Initialize Q-values for all action pairs
            self.belief_traker[state_key] = np.ones(5)  # Initialize belief tracker with uniform distribution
        return state_key

    def choose_action(self, state: tuple) -> str:
        """
        Choose an action based on the current state.

        Args:
            state (tuple): The current state of the environment.
        Returns:
            str: The chosen action for the agent.
        """
        state_key = self._get_or_create_state(state)
        if self.rng.random() < self.epsilon:
            # Exploration: choose a random action
            my_action = str(self.rng.choice(list(self.actions.keys())))
        else:
            # Exploitation: choose the best action based on Q-values
            q_values = self.q_table[state_key]
            # Convert belief to a probability distribution
            belief = self.belief_traker[state_key]
            belief = belief / np.sum(belief)
            expected_values = np.dot(q_values, belief)  # Calculate expected values for each action
            max_val = np.max(expected_values)
            best_action_index = int(self.rng.choice(np.flatnonzero(np.isclose(expected_values, max_val))))
            my_action = list(self.actions.keys())[best_action_index]
        return my_action
    
    def update(self, state: tuple, my_action: str, opp_action: str, my_reward: int, opp_reward: int, next_state: tuple, done: bool):
        """
        Update the Q-table and belief tracker based on the observed transition.

        Args:
            state (tuple): The previous state of the environment.
            my_action (str): The action taken by the current player.
            opp_action (str): The action taken by the opponent.
            my_reward (int): The reward received by the current player.
            opp_reward (int): The reward received by the opponent.
            next_state (tuple): The new state of the environment after taking the actions.
            done (bool): Whether the episode is done.
        """
        state_key = self._get_or_create_state(state)
        next_state_key = self._get_or_create_state(next_state)

        # Compute the next value based on the next state and the belief tracker
        if done:
            next_value = 0.0
        else:
            q_values_next = self.q_table[next_state_key]
            belief_next = self.belief_traker[next_state_key]
            belief_next_prob = belief_next / np.sum(belief_next)
            expected_values_next = np.dot(q_values_next, belief_next_prob)
            next_value = float(np.max(expected_values_next))

        a_idx = self.actions[my_action]
        o_idx = self.actions[opp_action]

        self.q_table[state_key][a_idx, o_idx] = (
            (1.0 - self.alpha) * self.q_table[state_key][a_idx, o_idx]
            + self.alpha * (my_reward + self.gamma * next_value)
        )

        # Update empirical count of opponent's action in the current state
        self.belief_traker[state_key][o_idx] += 1.0


class RandomAgent:
    """
    A simple agent that chooses actions randomly.
    """
    def __init__(self, rng: np.random.Generator | None = None):
        self.actions = ["0", "N", "S", "E", "W"]
        self.epsilon = 0.0  # Not used for RandomAgent
        self.alpha = 0.0    # Not used for RandomAgent
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_action(self, state: tuple) -> str:
        """
        Choose a random action.

        Args:
            state (tuple): The current state of the environment.
        Returns:
            str: The chosen action for the agent.
        """
        return np.random.choice(self.actions)

    def update(self, state: tuple, my_action: str, opp_action: str, my_reward: int, opp_reward: int, next_state: tuple, done: bool):
        """
        RandomAgent does not learn, so this method does nothing.

        Args:
            state (tuple): The previous state of the environment.
            my_action (str): The action taken by the current player.
            opp_action (str): The action taken by the opponent.
            my_reward (int): The reward received by the current player.
            opp_reward (int): The reward received by the opponent.
            next_state (tuple): The new state of the environment after taking the actions.
            done (bool): Whether the episode is done.
        """
        pass