"""
CUSTOM REINFORCEMENT LEARNING PROBLEM: IMAGE-BASED RACE TRACK CHALLENGE

PROBLEM DESCRIPTION:
An autonomous race car learns to navigate real race tracks loaded from images.
The system converts satellite imagery or track maps into a navigable grid where
the agent learns optimal racing lines through Q-learning.

CENTRAL VALLEY CONNECTION:
This demonstrates autonomous vehicle navigation applicable to California's
agricultural regions where autonomous tractors and vehicles operate.

IMAGE REQUIREMENTS:
1. track_map.png - Simple black/white track map OR
2. laguna_seca.png - Satellite image of Laguna Seca raceway

IMAGE FORMAT:
- Any size (will be resized to max 50x50 for computational efficiency)
- PNG or JPG format
- Color coding:
  * Black (0,0,0) = Walls/barriers (#)
  * White (255,255,255) = Safe track (T)
  * Green (0,255,0) = Start position (S)
  * Red (255,0,0) = Finish line (F)
  * For satellite images: Dark pixels = barriers, Light pixels = track

STATE SPACE:
- Position: (row, col) on track grid
- Velocity: 3 levels (SLOW=0, MEDIUM=1, FAST=2)
- Total states: (rows × cols × 3 velocities)

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
- Time penalty: -0.1 per step (encourages speed)
- Velocity bonus: +0.5 per velocity level (encourages fast driving)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================================
# STEP 1: IMAGE LOADING AND PROCESSING
# ============================================================================
print("=" * 70)
print("IMAGE-BASED RACE TRACK Q-LEARNING")
print("=" * 70)

# Image file to load (try these in order)
image_files = ['laguna_seca.png', 'track_map.png', 'racetrack.png', 'track.jpg']

# Find first available image
image_path = None
for img_file in image_files:
    if os.path.exists(img_file):
        image_path = img_file
        break

# If no image found, create a default track
if image_path is None:
    print("\nNO IMAGE FOUND - Creating default track")
    print("To use custom track, provide one of these images:")
    for img_file in image_files:
        print(f"  - {img_file}")
    print("\nImage color guide:")
    print("  BLACK (0,0,0) = Walls/barriers")
    print("  WHITE (255,255,255) = Safe track")
    print("  GREEN (0,255,0) = Start position")
    print("  RED (255,0,0) = Finish line")
    print("\nUsing default track layout...")
    
    # Default track map
    track_map = [
        ['F', 'F', 'F', 'T', 'T', 'T', 'T', 'F', 'F', 'F'],
        ['#', '#', 'T', 'T', 'T', 'T', 'T', 'T', '#', '#'],
        ['#', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', '#'],
        ['#', 'T', 'T', '#', '#', '#', 'T', 'T', 'T', '#'],
        ['#', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', '#'],
        ['#', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', '#'],
        ['#', '#', 'T', 'T', 'T', 'T', 'T', 'T', '#', '#'],
        ['#', '#', '#', 'T', 'T', 'T', 'T', '#', '#', '#'],
        ['#', '#', 'T', 'T', 'T', 'T', 'T', 'T', '#', '#'],
        ['#', '#', 'S', 'T', 'T', 'T', 'T', 'T', '#', '#'],
    ]
    
else:
    print(f"\nLOADING TRACK FROM IMAGE: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    print(f"Original image size: {img.size}")
    
    # Convert to RGB if needed
    img = img.convert('RGB')
    
    # Resize if too large (max 50x50 for computational efficiency)
    max_size = 50
    if img.size[0] > max_size or img.size[1] > max_size:
        # Maintain aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        print(f"Resized to: {img.size}")
    
    # Convert to numpy array
    img_array = np.array(img)
    height, width = img_array.shape[0], img_array.shape[1]
    
    print(f"Processing {width}x{height} track grid...")
    
    # Initialize track map
    track_map = []
    start_positions = []
    finish_positions = []
    
    # Convert pixels to track elements
    for row in range(height):
        track_row = []
        for col in range(width):
            # Get RGB values
            r, g, b = img_array[row, col]
            
            # Classify pixel based on color
            # Pure black = wall
            if r < 50 and g < 50 and b < 50:
                cell = '#'
            
            # Green = start (tolerance for greenish pixels)
            elif g > 200 and r < 100 and b < 100:
                cell = 'S'
                start_positions.append((row, col))
            
            # Red = finish (tolerance for reddish pixels)
            elif r > 200 and g < 100 and b < 100:
                cell = 'F'
                finish_positions.append((row, col))
            
            # Light pixels = track (white or light colored)
            elif r > 150 and g > 150 and b > 150:
                cell = 'T'
            
            # Medium-dark pixels = also track (for satellite images)
            elif r > 100 or g > 100 or b > 100:
                cell = 'T'
            
            # Very dark = wall
            else:
                cell = '#'
            
            track_row.append(cell)
        
        track_map.append(track_row)
    
    # If no start/finish found, add them
    if len(start_positions) == 0:
        # Put start at bottom center
        start_row = height - 1
        start_col = width // 2
        track_map[start_row][start_col] = 'S'
        start_positions.append((start_row, start_col))
        print("No start position in image - added at bottom center")
    
    if len(finish_positions) == 0:
        # Put finish at top center
        finish_row = 0
        for col in range(width):
            if track_map[finish_row][col] == 'T':
                track_map[finish_row][col] = 'F'
                finish_positions.append((finish_row, col))
        if len(finish_positions) == 0:
            track_map[0][width // 2] = 'F'
            finish_positions.append((0, width // 2))
        print("No finish line in image - added at top")
    
    print(f"Found {len(start_positions)} start position(s)")
    print(f"Found {len(finish_positions)} finish position(s)")

# Print track
print("\nRACE TRACK LAYOUT:")
print("S = Start | F = Finish | T = Track | # = Wall")
print("-" * 70)
for row in track_map:
    print(" ".join(row))
print("-" * 70)

# ============================================================================
# STEP 2: ENVIRONMENT SETUP
# ============================================================================
# Find start position
start_row = None
start_col = None
for row_idx in range(len(track_map)):
    for col_idx in range(len(track_map[row_idx])):
        if track_map[row_idx][col_idx] == 'S':
            start_row = row_idx
            start_col = col_idx
            break
    if start_row is not None:
        break

# If still no start, use bottom-left safe position
if start_row is None:
    for row_idx in range(len(track_map) - 1, -1, -1):
        for col_idx in range(len(track_map[row_idx])):
            if track_map[row_idx][col_idx] in ['T', 'F']:
                start_row = row_idx
                start_col = col_idx
                track_map[start_row][start_col] = 'S'
                break
        if start_row is not None:
            break

# Track dimensions
num_rows = len(track_map)
num_cols = len(track_map[0])

# Velocity levels
SLOW = 0
MEDIUM = 1
FAST = 2
num_velocity_levels = 3

# State space: position × velocity
state_space_size = num_rows * num_cols * num_velocity_levels

# Action space
ACCELERATE = 0
COAST = 1
BRAKE = 2
TURN_LEFT = 3
TURN_RIGHT = 4
action_space_size = 5
action_names = ['ACCELERATE', 'COAST', 'BRAKE', 'TURN_LEFT', 'TURN_RIGHT']

print(f"\nENVIRONMENT CONFIGURATION:")
print(f"Track size: {num_rows} × {num_cols}")
print(f"State space: {state_space_size} states")
print(f"Action space: {action_space_size} actions")
print(f"Start position: ({start_row}, {start_col})")

# ============================================================================
# STEP 3: STATE ENCODING/DECODING
# ============================================================================
def encode_state(row, col, velocity):
    return (row * num_cols * num_velocity_levels) + (col * num_velocity_levels) + velocity

def decode_state(state_id):
    velocity = state_id % num_velocity_levels
    remaining = state_id // num_velocity_levels
    col = remaining % num_cols
    row = remaining // num_cols
    return row, col, velocity

# ============================================================================
# STEP 4: ENVIRONMENT DYNAMICS
# ============================================================================
def step(current_state, action):
    row, col, velocity = decode_state(current_state)
    reward = -0.1
    reward += velocity * 0.5
    
    new_velocity = velocity
    if action == ACCELERATE:
        new_velocity = min(FAST, velocity + 1)
    elif action == BRAKE:
        new_velocity = max(SLOW, velocity - 1)
    
    row_change = 0
    col_change = 0
    
    if action == ACCELERATE or action == COAST or action == BRAKE:
        row_change = -2 if new_velocity == FAST else -1
        col_change = 0
    elif action == TURN_LEFT:
        row_change = -2 if velocity == FAST else -1
        col_change = -1
    elif action == TURN_RIGHT:
        row_change = -2 if velocity == FAST else -1
        col_change = 1
    
    new_row = row + row_change
    new_col = col + col_change
    
    if new_row < 0:
        new_row = 0
    if new_row >= num_rows:
        new_row = num_rows - 1
    if new_col < 0:
        new_col = 0
    if new_col >= num_cols:
        new_col = num_cols - 1
    
    cell_type = track_map[new_row][new_col]
    done = False
    
    if cell_type == '#':
        reward = -50
        done = True
        new_row = start_row
        new_col = start_col
        new_velocity = SLOW
    elif cell_type == 'F':
        reward = 100
        done = True
    elif cell_type == 'T' or cell_type == 'S':
        rows_advanced = row - new_row
        reward += rows_advanced * 1.0
    
    next_state = encode_state(new_row, new_col, new_velocity)
    return next_state, reward, done

def reset_environment():
    return encode_state(start_row, start_col, SLOW)

# ============================================================================
# STEP 5: INITIALIZE Q-TABLE
# ============================================================================
q_table = np.zeros((state_space_size, action_space_size))
print(f"\nInitialized Q-table: {q_table.shape}")

# ============================================================================
# STEP 6: HYPERPARAMETERS
# ============================================================================
alpha = 0.1
gamma = 0.95
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
num_episodes = 15000
max_steps_per_episode = 200

print("\n" + "=" * 70)
print("HYPERPARAMETERS")
print("=" * 70)
print(f"Learning rate (α): {alpha}")
print(f"Discount factor (γ): {gamma}")
print(f"Training episodes: {num_episodes}")
print("=" * 70)

# ============================================================================
# STEP 7: TRAINING LOOP
# ============================================================================
rewards_per_episode = []
epsilon_values = []
success_rate_window = []

epsilon = epsilon_start

print("\n" + "=" * 70)
print("STARTING TRAINING...")
print("=" * 70)

for episode in range(num_episodes):
    state = reset_environment()
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        if np.random.random() < epsilon:
            action = np.random.randint(0, action_space_size)
        else:
            action = np.argmax(q_table[state, :])
        
        next_state, reward, done = step(state, action)
        
        current_q = q_table[state, action]
        max_future_q = np.max(q_table[next_state, :])
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        q_table[state, action] = new_q
        
        state = next_state
        total_reward += reward
        steps += 1
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)
    epsilon_values.append(epsilon)
    
    if episode >= 100:
        recent_successes = sum(1 for r in rewards_per_episode[-100:] if r > 50)
        success_rate = recent_successes / 100.0
        success_rate_window.append(success_rate)
    
    if (episode + 1) % 1000 == 0:
        recent_success_rate = sum(1 for r in rewards_per_episode[-100:] if r > 50) / 100.0
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"Episode {episode + 1}/{num_episodes} | Success: {recent_success_rate:.2%} | "
              f"Avg Reward: {avg_reward:.1f} | ε: {epsilon:.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED!")
print("=" * 70)

# ============================================================================
# STEP 8: EVALUATION
# ============================================================================
print("\nEvaluating trained agent...")

test_episodes = 100
test_successes = 0

for episode in range(test_episodes):
    state = reset_environment()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        action = np.argmax(q_table[state, :])
        next_state, reward, done = step(state, action)
        state = next_state
        total_reward += reward
        steps += 1
    
    if total_reward > 50:
        test_successes += 1

final_success_rate = test_successes / test_episodes
print(f"\nFinal Success Rate: {final_success_rate:.2%}")
print(f"Successful Completions: {test_successes}/{test_episodes}")

# ============================================================================
# STEP 9: DEMONSTRATION
# ============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATING TRAINED AGENT")
print("=" * 70)

for demo in range(2):
    print(f"\n{'='*70}")
    print(f"DEMO RUN {demo + 1}")
    print(f"{'='*70}")
    
    state = reset_environment()
    done = False
    steps = 0
    total_reward = 0
    path = []
    
    while not done and steps < max_steps_per_episode:
        row, col, velocity = decode_state(state)
        path.append((row, col))
        
        action = np.argmax(q_table[state, :])
        next_state, reward, done = step(state, action)
        
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Steps: {steps}")
    print(f"Status: {'SUCCESS' if total_reward > 50 else 'FAILED'}")

# ============================================================================
# STEP 10: VISUALIZE POLICY
# ============================================================================
print("\n" + "=" * 70)
print("OPTIMAL POLICY (at MEDIUM velocity)")
print("=" * 70)

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
            state = encode_state(row, col, MEDIUM)
            best_action = np.argmax(q_table[state, :])
            action_symbols = ['ACC', 'CST', 'BRK', ' ↙ ', ' ↘ ']
            row_str += f"{action_symbols[best_action]}"
    
    print(row_str)

print("\nLegend: ACC=Accelerate, CST=Coast, BRK=Brake, ↙=Left, ↘=Right")

# ============================================================================
# STEP 11: PLOTS
# ============================================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS...")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(success_rate_window, color='blue', linewidth=2)
axes[0].axhline(y=final_success_rate, color='red', linestyle='--')
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Success Rate', fontsize=12)
axes[0].set_title('Training Progress: Success Rate', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epsilon_values, color='green', linewidth=2)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Epsilon', fontsize=12)
axes[1].set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('racetrack_training_results.png', dpi=300, bbox_inches='tight')
print("Saved: racetrack_training_results.png")
plt.show()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)