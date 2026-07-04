from environment import GridWorld
from agent import QLearningAgent


def train_agent(
    grid_size: int,
    num_obstacles: int,
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
):
    """
    Train a Q-Learning agent on the GridWorld environment.

    Returns:
        dict: Training results including the trained agent,
        environment, rewards, epsilon history and final statistics.
    """

    # Create environment
    env = GridWorld(grid_size, num_obstacles)

    # Total states and actions
    num_states = grid_size * grid_size
    num_actions = 4

    # Create agent
    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
    )

    episode_rewards = []
    epsilon_history = []

    # Prevent infinite episodes
    max_steps = grid_size * grid_size * 2

    # ==========================
    # Training Loop
    # ==========================
    for episode in range(num_episodes):

        position = env.reset()
        state = env.state_to_index(position)

        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:

            action = agent.choose_action(state)

            next_position, reward, done = env.step(action)

            next_state = env.state_to_index(next_position)

            agent.update_q_table(
                state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        epsilon_history.append(agent.epsilon)

        agent.decay_epsilon()

    return {
    "agent": agent,
    "environment": env,
    "episode_rewards": episode_rewards,
    "epsilon_history": epsilon_history,
    "q_table": agent.q_table,
    "final_epsilon": agent.epsilon,
    "total_episodes": num_episodes,
    "final_reward": episode_rewards[-1],
    "num_obstacles": num_obstacles,
    }   