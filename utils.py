import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Reward Plot
# -----------------------------
def plot_rewards(episode_rewards):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        episode_rewards,
        label="Episode Reward",
        linewidth=2,
    )

    ax.set_title("Episode Rewards During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(True)
    ax.legend()

    return fig


# -----------------------------
# Epsilon Plot
# -----------------------------
def plot_epsilon(epsilon_history):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        epsilon_history,
        color="orange",
        linewidth=2,
        label="Epsilon",
    )

    ax.set_title("Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True)
    ax.legend()

    return fig


# -----------------------------
# Policy Symbols
# -----------------------------
ACTION_SYMBOLS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}


# -----------------------------
# Extract Policy
# -----------------------------
def extract_policy(agent, env):
    policy = []

    # safety check
    if not hasattr(agent, "q_table"):
        raise ValueError("Agent does not have a q_table")

    for row in range(env.n):
        current_row = []

        for col in range(env.n):
            position = (row, col)

            if position == env.start:
                current_row.append("S")

            elif position == env.goal:
                current_row.append("G")

            elif position in env.obstacles:
                current_row.append("X")

            else:
                state = env.state_to_index(position)

                q_values = agent.q_table[state]

                # safety: handle uninitialized or wrong-shaped q-table
                best_action = int(np.argmax(q_values))

                current_row.append(ACTION_SYMBOLS.get(best_action, "?"))

        policy.append(current_row)

    return policy


# -----------------------------
# Policy Display
# -----------------------------
def display_policy(policy):
    return pd.DataFrame(policy)


# -----------------------------
# Q-table Display
# -----------------------------
def display_q_table(q_table):
    columns = ["UP", "DOWN", "LEFT", "RIGHT"]

    q_table = np.array(q_table)

    # safety check for shape mismatch
    if q_table.ndim != 2 or q_table.shape[1] != 4:
        raise ValueError(
            f"Unexpected Q-table shape: {q_table.shape}. Expected (states, 4)."
        )

    return pd.DataFrame(q_table, columns=columns)


# -----------------------------
# Test Agent
# -----------------------------
def test_agent(agent, env):
    path = []

    position = env.reset()
    path.append(position)

    done = False
    max_steps = env.n * env.n * 2
    steps = 0

    while not done and steps < max_steps:

        state = env.state_to_index(position)

        q_values = agent.q_table[state]

        action = int(np.argmax(q_values))

        next_position, reward, done = env.step(action)

        path.append(next_position)

        position = next_position
        steps += 1

    return path, done


# -----------------------------
# Grid Display
# -----------------------------
def create_grid_display(env, agent_position):
    grid = []

    for row in range(env.n):
        current_row = []

        for col in range(env.n):
            position = (row, col)

            if position == agent_position:
                current_row.append("🔵")

            elif position == env.start:
                current_row.append("🟢")

            elif position == env.goal:
                current_row.append("🏁")

            elif position in env.obstacles:
                current_row.append("⬛")

            else:
                current_row.append("⬜")

        grid.append(" ".join(current_row))

    return "\n".join(grid)