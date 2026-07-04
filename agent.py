import random
import numpy as np


class QLearningAgent:
    """
    Q-Learning agent that learns an optimal policy using the
    Bellman equation and an epsilon-greedy exploration strategy.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float,
        epsilon_min: float = 0.01,
    ):

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((num_states, num_actions), dtype=float)

    def choose_action(self, state: int) -> int:
        """
        Choose an action using the epsilon-greedy strategy.
        """

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[state])

        return action

    def update_q_table(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """
        Update the Q-value using the Bellman equation.
        """

        current_q = self.q_table[state][action]

        max_next_q_value = (
            np.max(self.q_table[next_state])
            if not done
            else 0
        )

        target = reward + self.gamma * max_next_q_value

        td_error = target - current_q

        new_q_value = current_q + self.alpha * td_error

        self.q_table[state, action] = new_q_value

    def decay_epsilon(self) -> None:
        """
        Decay the exploration rate after each episode.
        """

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)