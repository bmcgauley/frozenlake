import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: CREATE ENVIRONMENT
# ============================================================================
# Create 4x4 FrozenLake environment with slippery surface

# Create the FrozenLake environment (deterministic version for initial setup)
# env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode='ansi') 

# Get environment information
state_space_size = env.observation_space.n  # 16 states (0-15)
action_space_size = env.action_space.n      # 4 actions (LEFT, DOWN, RIGHT, UP)



print("=" * 60)
print("FROZENLAKE Q-LEARNING SETUP")
print("=" * 60)
print(f"State Space Size: {state_space_size}")
print(f"Action Space Size: {action_space_size}")
print(f"Actions: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP")
print("=" * 60)

# ============================================================================
# STEP 2: INITIALIZE Q-TABLE (THE BRAIN OF THE AGENT)
# ============================================================================
# Q-table stores the value of taking each action in each state
# Shape: [states, actions] = [16, 4]
# Initial values: All zeros (agent knows nothing at start)
q_table = np.zeros((state_space_size, action_space_size))

print("\nInitial Q-Table (all zeros - agent hasn't learned yet):")
print(q_table)

# ============================================================================
# STEP 3: SET HYPERPARAMETERS
# ============================================================================
# Alpha: Learning rate (how much to update Q-values each step)
# Higher = faster learning but less stable; Lower = slower but more stable
alpha = 0.1

# Gamma: Discount factor (how much to value future rewards)
# Close to 1 = care about long-term rewards; Close to 0 = only care about immediate rewards
gamma = 0.99

# Epsilon: Exploration rate (probability of taking random action)
# Start high (explore a lot) and decay (exploit more as we learn)
epsilon_start = 1.0      # Start with 100% exploration
epsilon_min = 0.01       # Minimum exploration (always explore 1%)
epsilon_decay = 0.995    # Decay rate per episode

# Training parameters
num_episodes = 10000     # Number of games to play during training
max_steps_per_episode = 100  # Maximum steps before giving up

print("\n" + "=" * 60)
print("HYPERPARAMETERS")
print("=" * 60)
print(f"Learning Rate (alpha): {alpha}")
print(f"Discount Factor (gamma): {gamma}")
print(f"Initial Epsilon: {epsilon_start}")
print(f"Minimum Epsilon: {epsilon_min}")
print(f"Epsilon Decay: {epsilon_decay}")
print(f"Training Episodes: {num_episodes}")
print("=" * 60)

# ============================================================================
# STEP 4: TRAINING LOOP (Q-LEARNING ALGORITHM)
# ============================================================================
# Track performance metrics
rewards_per_episode = []
epsilon_values = []
success_rate_window = []

# Initialize epsilon
epsilon = epsilon_start

print("\n" + "=" * 60)
print("STARTING TRAINING...")
print("=" * 60)

# Loop through each episode (game)
for episode in range(num_episodes):
    
    # Reset environment to starting state
    state, info = env.reset()
    
    # Track total reward for this episode
    total_reward = 0
    
    # Flag to track if episode is done
    done = False
    
    # Track steps taken in this episode
    steps = 0
    
    # Play one episode (until goal/hole reached or max steps)
    while not done and steps < max_steps_per_episode:
        
        # ================================================================
        # STEP 4A: EPSILON-GREEDY ACTION SELECTION
        # ================================================================
        # Generate random number between 0 and 1
        random_value = np.random.random()
        
        # If random value < epsilon, EXPLORE (random action)
        if random_value < epsilon:
            action = env.action_space.sample()  # Random action
        
        # Otherwise, EXPLOIT (use best known action from Q-table)
        else:
            action = np.argmax(q_table[state, :])  # Action with highest Q-value
        
        # ================================================================
        # STEP 4B: TAKE ACTION IN ENVIRONMENT
        # ================================================================
        # Execute action and observe result
        next_state, reward, done, truncated, info = env.step(action)
        
        # ================================================================
        # STEP 4C: Q-TABLE UPDATE (BELLMAN EQUATION)
        # ================================================================
        # Current Q-value for this state-action pair
        current_q_value = q_table[state, action]
        
        # Maximum Q-value for next state (best possible future value)
        max_future_q_value = np.max(q_table[next_state, :])
        
        # Calculate new Q-value using Bellman equation
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q_value = current_q_value + alpha * (reward + gamma * max_future_q_value - current_q_value)
        
        # Update Q-table with new value
        q_table[state, action] = new_q_value
        
        # ================================================================
        # STEP 4D: MOVE TO NEXT STATE
        # ================================================================
        state = next_state
        total_reward += reward
        steps += 1
    
    # ================================================================
    # STEP 4E: DECAY EPSILON (REDUCE EXPLORATION OVER TIME)
    # ================================================================
    # Gradually reduce epsilon so agent explores less and exploits more
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # ================================================================
    # STEP 4F: TRACK METRICS
    # ================================================================
    rewards_per_episode.append(total_reward)
    epsilon_values.append(epsilon)
    
    # Calculate success rate over last 100 episodes
    if episode >= 100:
        recent_successes = sum(rewards_per_episode[-100:])
        success_rate = recent_successes / 100.0
        success_rate_window.append(success_rate)
    
    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        recent_success_rate = sum(rewards_per_episode[-100:]) / 100.0
        print(f"Episode {episode + 1}/{num_episodes} | Success Rate (last 100): {recent_success_rate:.2%} | Epsilon: {epsilon:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)

# ============================================================================
# STEP 5: EVALUATE TRAINED AGENT
# ============================================================================
print("\nEvaluating trained agent over 100 test episodes...")

# Test the trained agent
test_episodes = 100
test_rewards = []

for episode in range(test_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        # Use ONLY exploitation (no exploration) - take best action
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1
    
    test_rewards.append(total_reward)

# Calculate final success rate
final_success_rate = sum(test_rewards) / test_episodes

print(f"\nFinal Success Rate: {final_success_rate:.2%}")
print(f"Total Successful Episodes: {sum(test_rewards)}/{test_episodes}")

# ============================================================================
# STEP 6: DISPLAY LEARNED Q-TABLE
# ============================================================================
print("\n" + "=" * 60)
print("LEARNED Q-TABLE (Agent's Knowledge)")
print("=" * 60)
print("Each row = state (0-15), Each column = action (LEFT, DOWN, RIGHT, UP)")
print(q_table)
print("=" * 60)

# ============================================================================
# STEP 7: DEMONSTRATE TRAINED AGENT (VISUAL)
# ============================================================================
print("\n" + "=" * 60)
print("DEMONSTRATING TRAINED AGENT (3 EPISODES)")
print("=" * 60)

# Create new environment with human render mode for visualization
env_demo = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='ansi')

for demo_episode in range(3):
    print(f"\n--- Demo Episode {demo_episode + 1} ---")
    state, info = env_demo.reset()
    done = False
    steps = 0
    total_reward = 0
    
    print(env_demo.render())
    
    while not done and steps < max_steps_per_episode:
        # Use best action from Q-table
        action = np.argmax(q_table[state, :])
        
        # Action names for readability
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        print(f"Step {steps + 1}: State={state}, Action={action_names[action]}")
        
        next_state, reward, done, truncated, info = env_demo.step(action)
        print(env_demo.render())
        
        state = next_state
        total_reward += reward
        steps += 1
        
        time.sleep(0.3)  # Pause for visualization
    
    if total_reward == 1:
        print("SUCCESS! Reached the goal!")
    else:
        print("FAILED! Fell into a hole.")

env_demo.close()

# ============================================================================
# STEP 8: PLOT TRAINING METRICS
# ============================================================================
print("\n" + "=" * 60)
print("GENERATING TRAINING PLOTS...")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Success rate over time (rolling average)
axes[0].plot(success_rate_window, color='blue', linewidth=2)
axes[0].axhline(y=final_success_rate, color='red', linestyle='--', label=f'Final Rate: {final_success_rate:.2%}')
axes[0].set_xlabel('Episode (starting from episode 100)', fontsize=12)
axes[0].set_ylabel('Success Rate (last 100 episodes)', fontsize=12)
axes[0].set_title('Q-Learning Training Progress: Success Rate Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Epsilon decay
axes[1].plot(epsilon_values, color='green', linewidth=2)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
axes[1].set_title('Exploration vs Exploitation: Epsilon Decay', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('frozenlake_training_results.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'frozenlake_training_results.png'")
plt.show()

# ============================================================================
# STEP 9: DISPLAY OPTIMAL POLICY
# ============================================================================
print("\n" + "=" * 60)
print("OPTIMAL POLICY (Best Action for Each State)")
print("=" * 60)

# Extract optimal policy from Q-table
optimal_policy = np.argmax(q_table, axis=1)
action_symbols = ['←', '↓', '→', '↑']

# Display as 4x4 grid
print("\nPolicy Grid (4x4):")
print("S = Start, F = Frozen, H = Hole, G = Goal")
print("Arrows show best action from each state:\n")

map_grid = [
    ['S', 'H', 'G', 'H'],
    ['F', 'H', 'F', 'F'],
    ['F', 'H', 'H', 'F'],
    ['F', 'F', 'F', 'F']
]

for row in range(4):
    policy_row = []
    for col in range(4):
        state = row * 4 + col
        cell = map_grid[row][col]
        
        if cell == 'H' or cell == 'G':
            policy_row.append(f" {cell} ")
        else:
            arrow = action_symbols[optimal_policy[state]]
            policy_row.append(f" {arrow} ")
    
    print(" ".join(policy_row))

print("\n" + "=" * 60)
print("TRAINING COMPLETE! Agent is ready.")
print("=" * 60)




















done = False 
state, info = env.reset() 
env.render() 
 
while not done: 
    action = env.action_space.sample()  # Pick a random action 
    state, reward, done, truncated, info = env.step(action) 
    env.render() 
    time.sleep(1)  # Pause to see each move 
    if done: 
        print(f"Episode finished. Reward: {reward}") 
        break