"""
CUSTOM REINFORCEMENT LEARNING PROBLEM: RACE TRACK CHALLENGE

PROBLEM DESCRIPTION:
An autonomous race car must learn to navigate a race track from START to FINISH
while avoiding walls and optimizing speed. The car has limited visibility and
must learn through trial and error to find the optimal racing line.

ENVIRONMENT SPECIFICATIONS:
- 10x10 grid race track
- Car starts at position (9, 0) with zero velocity
- Goal: Reach finish line at top row (row 0)
- Walls (#): Crash zones that end episode with penalty
- Track (T): Safe racing surface
- Start (S): Starting position
- Finish (F): Goal positions

STATE SPACE:
- Position: (row, col) on 10x10 grid = 100 positions
- Velocity: 3 levels (SLOW=0, MEDIUM=1, FAST=2)
- Total states: 100 positions × 3 velocities = 300 states

ACTION SPACE:
5 discrete actions:
- 0: ACCELERATE (increase velocity, move forward)
- 1: COAST (maintain velocity, move forward)
- 2: BRAKE (decrease velocity, move forward)
- 3: TURN_LEFT (move forward-left)
- 4: TURN_RIGHT (move forward-right)

REWARD STRUCTURE:
- Reach finish: +100 points
- Forward progress: +1 per row advanced
- Crash into wall: -50 points (episode ends)
- Each time step: -0.1 (time penalty, encourages speed)
- Higher velocity bonus: +0.5 per velocity level (encourages fast driving)

PHYSICS:
- Car always moves in direction based on action
- Velocity affects how far car moves (FAST moves 2 rows, others 1 row)
- Crashes reset episode

LEARNING OBJECTIVE:
Agent must learn to balance speed (fast = good) with safety (avoid walls).
Optimal policy navigates track quickly without crashing.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: DEFINE RACE TRACK MAP
# ============================================================================
print("=" * 70)
print("CUSTOM RACE TRACK Q-LEARNING PROBLEM")
print("=" * 70)

# Track legend:
# S = Start, F = Finish, T = Track (safe), # = Wall (crash)
# Track is 10 rows × 10 columns
track_map = [
    ['F', 'F', 'F', 'T', 'T', 'T', 'T', 'F', 'F', 'F'],  # Row 0: Finish line
    ['#', '#', 'T', 'T', 'T', 'T', 'T', 'T', '#', '#'],  # Row 1
    ['#', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', '#'],  # Row 2
    ['#', 'T', 'T', '#', '#', '#', 'T', 'T', 'T', '#'],  # Row 3: Narrow section
    ['#', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', '#'],  # Row 4
    ['#', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', '#'],  # Row 5
    ['#', '#', 'T', 'T', 'T', 'T', 'T', 'T', '#', '#'],  # Row 6: Wide turn
    ['#', '#', '#', 'T', 'T', 'T', 'T', '#', '#', '#'],  # Row 7: Narrow
    ['#', '#', 'T', 'T', 'T', 'T', 'T', 'T', '#', '#'],  # Row 8
    ['#', '#', 'S', 'T', 'T', 'T', 'T', 'T', '#', '#'],  # Row 9: Start
]

# Print track
print("\nRACE TRACK LAYOUT:")
print("S = Start | F = Finish | T = Track | # = Wall (crash)")
print("-" * 70)
for row in track_map:
    print(" ".join(row))
print("-" * 70)

# ============================================================================
# STEP 2: ENVIRONMENT SETUP
# ============================================================================
# Starting position
start_row = 9
start_col = 2

# Track dimensions
num_rows = len(track_map)
num_cols = len(track_map[0])

# Velocity levels
SLOW = 0
MEDIUM = 1
FAST = 2
num_velocity_levels = 3

# State space: position (100) × velocity (3) = 300 states
# State encoding: state_id = (row * num_cols * num_velocity_levels) + (col * num_velocity_levels) + velocity
state_space_size = num_rows * num_cols * num_velocity_levels

# Action space: 5 actions
ACCELERATE = 0
COAST = 1
BRAKE = 2
TURN_LEFT = 3
TURN_RIGHT = 4
action_space_size = 5

action_names = ['ACCELERATE', 'COAST', 'BRAKE', 'TURN_LEFT', 'TURN_RIGHT']

print(f"\nENVIRONMENT CONFIGURATION:")
print(f"Track size: {num_rows} × {num_cols}")
print(f"State space: {state_space_size} states (position × velocity)")
print(f"Action space: {action_space_size} actions")
print(f"Start position: ({start_row}, {start_col})")
print(f"Velocity levels: SLOW(0), MEDIUM(1), FAST(2)")

# ============================================================================
# STEP 3: STATE ENCODING/DECODING FUNCTIONS
# ============================================================================
# Convert (row, col, velocity) to single state ID
def encode_state(row, col, velocity):
    return (row * num_cols * num_velocity_levels) + (col * num_velocity_levels) + velocity

# Convert state ID back to (row, col, velocity)
def decode_state(state_id):
    velocity = state_id % num_velocity_levels
    remaining = state_id // num_velocity_levels
    col = remaining % num_cols
    row = remaining // num_cols
    return row, col, velocity

# ============================================================================
# STEP 4: ENVIRONMENT DYNAMICS (TRANSITION FUNCTION)
# ============================================================================
# Take action in environment, return next state and reward
def step(current_state, action):
    # Decode current state
    row, col, velocity = decode_state(current_state)
    
    # Initialize reward
    reward = -0.1  # Time penalty (encourages finishing quickly)
    
    # Add velocity bonus (encourages speed)
    reward += velocity * 0.5
    
    # Update velocity based on action
    new_velocity = velocity
    if action == ACCELERATE:
        new_velocity = min(FAST, velocity + 1)  # Increase velocity (cap at FAST)
    elif action == BRAKE:
        new_velocity = max(SLOW, velocity - 1)  # Decrease velocity (floor at SLOW)
    # COAST and turning actions maintain current velocity
    
    # Determine movement based on action
    row_change = 0
    col_change = 0
    
    if action == ACCELERATE or action == COAST or action == BRAKE:
        # Move forward (toward row 0)
        row_change = -2 if new_velocity == FAST else -1
        col_change = 0
    elif action == TURN_LEFT:
        # Move forward and left
        row_change = -2 if velocity == FAST else -1
        col_change = -1
    elif action == TURN_RIGHT:
        # Move forward and right
        row_change = -2 if velocity == FAST else -1
        col_change = 1
    
    # Calculate new position
    new_row = row + row_change
    new_col = col + col_change
    
    # Check boundaries (can't go off grid)
    if new_row < 0:
        new_row = 0
    if new_row >= num_rows:
        new_row = num_rows - 1
    if new_col < 0:
        new_col = 0
    if new_col >= num_cols:
        new_col = num_cols - 1
    
    # Check what cell we landed on
    cell_type = track_map[new_row][new_col]
    
    # Initialize done flag
    done = False
    
    # Handle different cell types
    if cell_type == '#':
        # Crashed into wall
        reward = -50
        done = True
        new_row = start_row  # Reset to start
        new_col = start_col
        new_velocity = SLOW
    elif cell_type == 'F':
        # Reached finish line
        reward = 100
        done = True
    elif cell_type == 'T' or cell_type == 'S':
        # Safe track - add progress reward
        rows_advanced = row - new_row
        reward += rows_advanced * 1.0  # +1 per row toward finish
    
    # Encode new state
    next_state = encode_state(new_row, new_col, new_velocity)
    
    return next_state, reward, done

# ============================================================================
# STEP 5: RESET ENVIRONMENT
# ============================================================================
def reset_environment():
    # Return to starting position with zero velocity
    return encode_state(start_row, start_col, SLOW)

# ============================================================================
# STEP 6: INITIALIZE Q-TABLE
# ============================================================================
# Q-table: [states × actions] = [300 × 5]
q_table = np.zeros((state_space_size, action_space_size))

print(f"\nInitialized Q-table shape: {q_table.shape}")
print("Q-table represents value of each action in each state")

# ============================================================================
# STEP 7: SET HYPERPARAMETERS
# ============================================================================
alpha = 0.1              # Learning rate
gamma = 0.95             # Discount factor (slightly lower - want immediate progress)
epsilon_start = 1.0      # Initial exploration rate
epsilon_min = 0.01       # Minimum exploration rate
epsilon_decay = 0.9995   # Slower decay for harder problem
num_episodes = 15000     # More episodes for complex environment
max_steps_per_episode = 100  # Maximum steps before timeout

print("\n" + "=" * 70)
print("HYPERPARAMETERS")
print("=" * 70)
print(f"Learning rate (α): {alpha}")
print(f"Discount factor (γ): {gamma}")
print(f"Initial epsilon: {epsilon_start}")
print(f"Epsilon decay: {epsilon_decay}")
print(f"Training episodes: {num_episodes}")
print(f"Max steps per episode: {max_steps_per_episode}")
print("=" * 70)

# ============================================================================
# STEP 8: TRAINING LOOP (Q-LEARNING)
# ============================================================================
rewards_per_episode = []
epsilon_values = []
success_rate_window = []
steps_per_episode = []

epsilon = epsilon_start

print("\n" + "=" * 70)
print("STARTING TRAINING...")
print("=" * 70)

# Training loop
for episode in range(num_episodes):
    
    # Reset to start state
    state = reset_environment()
    
    # Track episode metrics
    total_reward = 0
    done = False
    steps = 0
    
    # Run episode
    while not done and steps < max_steps_per_episode:
        
        # Epsilon-greedy action selection
        random_value = np.random.random()
        
        if random_value < epsilon:
            # Explore: random action
            action = np.random.randint(0, action_space_size)
        else:
            # Exploit: best action from Q-table
            action = np.argmax(q_table[state, :])
        
        # Take action in environment
        next_state, reward, done = step(state, action)
        
        # Q-learning update (Bellman equation)
        current_q = q_table[state, action]
        max_future_q = np.max(q_table[next_state, :])
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        q_table[state, action] = new_q
        
        # Move to next state
        state = next_state
        total_reward += reward
        steps += 1
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Track metrics
    rewards_per_episode.append(total_reward)
    epsilon_values.append(epsilon)
    steps_per_episode.append(steps)
    
    # Calculate success rate (reward > 50 means reached finish)
    if episode >= 100:
        recent_successes = sum(1 for r in rewards_per_episode[-100:] if r > 50)
        success_rate = recent_successes / 100.0
        success_rate_window.append(success_rate)
    
    # Print progress
    if (episode + 1) % 1000 == 0:
        recent_success_rate = sum(1 for r in rewards_per_episode[-100:] if r > 50) / 100.0
        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_steps = np.mean(steps_per_episode[-100:])
        print(f"Episode {episode + 1}/{num_episodes} | Success: {recent_success_rate:.2%} | "
              f"Avg Reward: {avg_reward:.1f} | Avg Steps: {avg_steps:.1f} | ε: {epsilon:.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED!")
print("=" * 70)

# ============================================================================
# STEP 9: EVALUATE TRAINED AGENT
# ============================================================================
print("\nEvaluating trained agent over 100 test episodes...")

test_episodes = 100
test_rewards = []
test_successes = 0

for episode in range(test_episodes):
    state = reset_environment()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        # Only exploit (no exploration)
        action = np.argmax(q_table[state, :])
        next_state, reward, done = step(state, action)
        state = next_state
        total_reward += reward
        steps += 1
    
    test_rewards.append(total_reward)
    if total_reward > 50:  # Successfully reached finish
        test_successes += 1

final_success_rate = test_successes / test_episodes
avg_test_reward = np.mean(test_rewards)

print(f"\nFinal Success Rate: {final_success_rate:.2%}")
print(f"Average Reward: {avg_test_reward:.1f}")
print(f"Successful Completions: {test_successes}/{test_episodes}")

# ============================================================================
# STEP 10: DEMONSTRATE TRAINED AGENT
# ============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATING TRAINED AGENT (3 EPISODES)")
print("=" * 70)

for demo in range(3):
    print(f"\n{'='*70}")
    print(f"DEMO RUN {demo + 1}")
    print(f"{'='*70}")
    
    state = reset_environment()
    done = False
    steps = 0
    total_reward = 0
    
    # Visualize starting position
    row, col, velocity = decode_state(state)
    velocity_names = ['SLOW', 'MEDIUM', 'FAST']
    print(f"\nStart: Position ({row},{col}), Velocity: {velocity_names[velocity]}")
    
    while not done and steps < max_steps_per_episode:
        # Get best action
        action = np.argmax(q_table[state, :])
        
        # Display action
        row, col, velocity = decode_state(state)
        print(f"\nStep {steps + 1}:")
        print(f"  Position: ({row},{col}) | Velocity: {velocity_names[velocity]}")
        print(f"  Action: {action_names[action]}")
        
        # Take action
        next_state, reward, done = step(state, action)
        
        # Update state
        state = next_state
        total_reward += reward
        steps += 1
        
        # Check result
        new_row, new_col, new_velocity = decode_state(next_state)
        cell = track_map[new_row][new_col]
        
        if cell == '#':
            print(f"  Result: CRASHED into wall at ({new_row},{new_col}) ❌")
        elif cell == 'F':
            print(f"  Result: FINISHED at ({new_row},{new_col}) ✓")
        else:
            print(f"  Result: Moved to ({new_row},{new_col}), {velocity_names[new_velocity]}")
        
        time.sleep(0.2)
    
    print(f"\nEpisode Result:")
    print(f"  Total Reward: {total_reward:.1f}")
    print(f"  Steps Taken: {steps}")
    print(f"  Status: {'SUCCESS' if total_reward > 50 else 'FAILED'}")

# ============================================================================
# STEP 11: VISUALIZE OPTIMAL POLICY
# ============================================================================
print("\n" + "=" * 70)
print("OPTIMAL POLICY VISUALIZATION")
print("=" * 70)
print("Showing best action for each position (at MEDIUM velocity)")

# Create policy visualization for MEDIUM velocity
print("\nPolicy at MEDIUM velocity:")
for row in range(num_rows):
    row_str = ""
    for col in range(num_cols):
        cell = track_map[row][col]
        
        if cell == '#':
            row_str += "  #  "
        elif cell == 'F':
            row_str += "  F  "
        elif cell == 'S':
            row_str += "  S  "
        else:
            # Get best action for this position at MEDIUM velocity
            state = encode_state(row, col, MEDIUM)
            best_action = np.argmax(q_table[state, :])
            
            # Use symbols for actions
            action_symbols = ['ACC', 'CST', 'BRK', '↙', '↘']
            row_str += f" {action_symbols[best_action]} "
    
    print(row_str)

print("\nLegend:")
print("ACC = Accelerate | CST = Coast | BRK = Brake")
print("↙ = Turn Left | ↘ = Turn Right")

# ============================================================================
# STEP 12: PLOT TRAINING METRICS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING TRAINING PLOTS...")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Plot 1: Success rate over time
axes[0].plot(success_rate_window, color='blue', linewidth=2)
axes[0].axhline(y=final_success_rate, color='red', linestyle='--', 
                label=f'Final: {final_success_rate:.1%}')
axes[0].set_xlabel('Episode (starting from 100)', fontsize=12)
axes[0].set_ylabel('Success Rate (last 100 episodes)', fontsize=12)
axes[0].set_title('Race Track Learning: Success Rate Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Average reward over time (smoothed)
window_size = 100
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
axes[1].plot(smoothed_rewards, color='green', linewidth=2)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel(f'Average Reward (window={window_size})', fontsize=12)
axes[1].set_title('Average Reward Progress', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Epsilon decay
axes[2].plot(epsilon_values, color='orange', linewidth=2)
axes[2].set_xlabel('Episode', fontsize=12)
axes[2].set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
axes[2].set_title('Exploration vs Exploitation Balance', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('racetrack_training_results.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'racetrack_training_results.png'")
plt.show()

print("\n" + "=" * 70)
print("RACE TRACK Q-LEARNING COMPLETE!")
print("=" * 70)
print("\nKEY ACHIEVEMENTS:")
print(f"✓ Trained agent over {num_episodes} episodes")
print(f"✓ Final success rate: {final_success_rate:.1%}")
print(f"✓ State space: {state_space_size} states (position × velocity)")
print(f"✓ Action space: {action_space_size} actions")
print(f"✓ Learned optimal racing policy")
print("=" * 70)