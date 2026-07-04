import streamlit as st
import time

from train import train_agent
import utils

# ======================================
# Page Config
# ======================================
st.set_page_config(
    page_title="GridWorld Q-Learning Simulator",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 GridWorld Q-Learning Simulator")
st.caption("An interactive Reinforcement Learning project built from scratch using Q-Learning.")

# ======================================
# Session State (IMPORTANT FIX)
# ======================================
if "results" not in st.session_state:
    st.session_state.results = None


# ======================================
# About Section
# ======================================
with st.expander("📘 About this Project", expanded=False):
    st.markdown("""
This project demonstrates how a **Q-Learning agent** learns to navigate a GridWorld environment.

- Q-Learning from scratch  
- ε-greedy policy  
- Bellman updates  
- Streamlit visualization  
""")

with st.expander("🧠 How Q-Learning Works", expanded=False):
    st.latex(r"Q(s,a)=Q(s,a)+\alpha[r+\gamma\max Q(s',a')-Q(s,a)]")


# ======================================
# Sidebar
# ======================================
st.sidebar.header("Training Parameters")

num_episodes = st.sidebar.slider("Episodes", 100, 5000, 1000, 100)
alpha = st.sidebar.slider("Alpha", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Gamma", 0.1, 0.99, 0.9)
epsilon = st.sidebar.slider("Epsilon", 0.1, 1.0, 1.0)
epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.90, 0.999, 0.995)
grid_size = st.sidebar.slider("Grid Size", 4, 10, 5)
num_obstacles = st.sidebar.slider("Obstacles", 0, 20, 5)

st.sidebar.markdown("---")
st.sidebar.info("Built by Shreya Maurya")


# ======================================
# TRAIN BUTTON (FIXED)
# ======================================
if st.button("🚀 Train Agent"):

    with st.spinner("Training agent..."):

        st.session_state.results = train_agent(
            grid_size=grid_size,
            num_obstacles=num_obstacles,
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
        )

    st.success("Training completed successfully!")


# ======================================
# SHOW RESULTS ONLY IF TRAINED
# ======================================
if st.session_state.results is not None:

    results = st.session_state.results

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Episode Reward Graph")
        st.pyplot(utils.plot_rewards(results["episode_rewards"]))

    with col2:
        st.subheader("Epsilon Decay Graph")
        st.pyplot(utils.plot_epsilon(results["epsilon_history"]))


    st.subheader("Learned Policy")
    policy = utils.extract_policy(results["agent"], results["environment"])
    st.dataframe(utils.display_policy(policy), use_container_width=True)


    st.subheader("Final Q-Table")
    st.dataframe(utils.display_q_table(results["q_table"]), use_container_width=True)


# ======================================
# TEST BUTTON (FIXED - OUTSIDE TRAIN BLOCK)
# ======================================
st.subheader("Test the Learned Agent")

if st.button("▶ Test Agent"):

    if st.session_state.results is None:
        st.warning("Train the agent first!")
    else:
        path, success = utils.test_agent(
            st.session_state.results["agent"],
            st.session_state.results["environment"],
        )

        placeholder = st.empty()

        for position in path:

            grid = utils.create_grid_display(
                st.session_state.results["environment"],
                position,
            )

            placeholder.text(grid)
            time.sleep(0.4)

        if success:
            st.success("🎉 Goal reached successfully!")
        else:
            st.error("❌ Agent could not reach the goal.")


# ======================================
# SUMMARY
# ======================================
if st.session_state.results is not None:

    results = st.session_state.results

    st.subheader("Training Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Episodes", results["total_episodes"])

    with col2:
        st.metric("Final Epsilon", f"{results['final_epsilon']:.4f}")

    with col3:
        st.metric("Grid Size", f"{grid_size} × {grid_size}")