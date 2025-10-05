# ================================
# FrozenLake Q-learning (with Step Playback)
# ================================
# Install (if needed):
#   pip install gymnasium
# Optional (only for 'human' windowed renders): pip install pygame

import gymnasium as gym
import numpy as np
import time
from typing import Tuple

# ---------------------------
# 1) Create the environment
# ---------------------------
# Slippery version (stochastic), as in typical labs:
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="ansi")

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Action space:", env.action_space)            # Discrete(4)
print("Observation space:", env.observation_space)  # Discrete(16)

# ---------------------------
# (Optional) Quick random run
# ---------------------------
def play_random_episode(env, sleep: float = 0.2):
    state, info = env.reset()
    done = False
    total_reward = 0
    print("\nRandom policy episode (for illustration):")
    print(env.render())
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        print(env.render())
        time.sleep(sleep)
        state = next_state
    print(f"Finished. Reward: {total_reward}")

# Uncomment to see a single random episode:
# play_random_episode(env)

# ---------------------------------------
# 2) Q-learning hyperparameters & memory
# ---------------------------------------
Q = np.zeros((n_states, n_actions), dtype=np.float32)

alpha = 0.1         # learning rate
gamma = 0.99        # discount factor
epsilon = 1.0       # starting exploration rate
epsilon_min = 0.01
epsilon_decay = 0.9995    # decay per episode (tune as desired)
episodes = 30000          # more episodes helps slippery lake converge
max_steps_per_episode = 200

rng = np.random.default_rng(42)  # reproducible exploration

def epsilon_greedy_action(state: int) -> int:
    if rng.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))

# ----------------------------
# 3) Training loop (Q-learning)
# ----------------------------
def train():
    global epsilon
    rewards_history = []
    for ep in range(episodes):
        state, info = env.reset()
        total_reward = 0

        for t in range(max_steps_per_episode):
            action = epsilon_greedy_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Q-learning update
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next * (0 if done else 1)
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * td_target

            state = next_state
            total_reward += reward
            if done:
                break

        # epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        # occasional training log
        if (ep + 1) % 5000 == 0:
            avg_last_5k = np.mean(rewards_history[-5000:])
            print(f"Episode {ep+1}/{episodes} | epsilon={epsilon:.3f} | "
                  f"Avg reward (last 5k): {avg_last_5k:.3f}")
    return rewards_history

print("\n=== Training Q-learning agent on FrozenLake-v1 (slippery) ===")
train_rewards = train()

# ---------------------------------
# 4) Derive greedy policy from Q(s)
# ---------------------------------
policy = np.argmax(Q, axis=1)  # shape: (n_states,)

# Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
action_to_arrow = {0: "←", 1: "↓", 2: "→", 3: "↑"}
action_to_name = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

def decode_map(env) -> np.ndarray:
    desc = env.unwrapped.desc  # bytes array of shape (nrow, ncol)
    grid = np.array([[c.decode("utf-8") for c in row] for row in desc])
    return grid

def print_policy_grid(env, policy):
    grid = decode_map(env)
    nrow, ncol = grid.shape
    out = []
    for s in range(n_states):
        r, c = divmod(s, ncol)
        cell = grid[r, c]
        if cell in ("S", "H", "G"):
            out.append(cell)
        else:
            out.append(action_to_arrow[int(policy[s])])
    print("\nLearned greedy policy (arrows). S=start, G=goal, H=hole:")
    for r in range(nrow):
        row_str = " ".join(out[r*ncol:(r+1)*ncol])
        print(row_str)

print_policy_grid(env, policy)

# ------------------------------------------------
# 5) Evaluate the learned policy over 100 episodes
# ------------------------------------------------
def run_greedy_episode(env, render: bool = False, sleep: float = 0.1) -> Tuple[float, int]:
    state, info = env.reset()
    total_reward, steps = 0.0, 0
    done = False

    while not done and steps < max_steps_per_episode:
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        if render:
            print(env.render())
            time.sleep(sleep)

        state = next_state
    return total_reward, steps

N_EVAL = 100
successes, total_steps = 0, 0

for _ in range(N_EVAL):
    r, steps = run_greedy_episode(env, render=False)
    successes += int(r > 0.0)
    total_steps += steps

success_rate = 100.0 * successes / N_EVAL
avg_steps = total_steps / N_EVAL

print(f"\n=== Evaluation over {N_EVAL} episodes ===")
print(f"Successes: {successes}/{N_EVAL} ({success_rate:.1f}%)")
print(f"Average steps per episode: {avg_steps:.1f}")

# ------------------------------------------------------------
# 6) NEW: Step-by-step playback of a single greedy-policy run
# ------------------------------------------------------------
def run_and_show_episode(env, Q, sleep: float = 0.5, max_steps: int = 100):
    """Run one greedy episode using the learned Q-table and show steps."""
    state, info = env.reset()
    done = False
    total_reward, steps = 0, 0

    print("\nAgent run (greedy policy):")
    print(env.render())  # initial map/frame

    while not done and steps < max_steps:
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Show step, action, and current frame
        print(f"\nStep {steps}: Action = {action} ({action_to_name[action]})")
        print(env.render())
        time.sleep(sleep)

        state = next_state

    print(f"\nEpisode finished after {steps} steps. "
          f"Reward = {total_reward} ({'SUCCESS' if total_reward > 0 else 'FAIL'})")

# Uncomment to watch a single run with the learned policy:
run_and_show_episode(env, Q, sleep=0.4, max_steps=100)
