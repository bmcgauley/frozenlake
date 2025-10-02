"""
LAGUNA SECA SATELLITE Q-LEARNING - FIXED COLOR DETECTION

CRITICAL FIX: Satellite track is DARK BLUE/GRAY, not black
Track detection uses proper color ranges for satellite imagery
Random start position ensures starting ON track
Circuit-based goals with lap counting
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAINING_IMAGE = 'img/laguna_seca.jpg'  # Satellite image

print("=" * 70)
print("LAGUNA SECA Q-LEARNING - SATELLITE TRACK")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD IMAGE
# ============================================================================
if not os.path.exists(TRAINING_IMAGE):
    print(f"\nERROR: Image not found: {TRAINING_IMAGE}")
    exit(1)

print(f"\nLOADING: {TRAINING_IMAGE}")
img = Image.open(TRAINING_IMAGE).convert('RGB')
print(f"Original size: {img.size}")

img_array = np.array(img)
height, width = img_array.shape[0], img_array.shape[1]
total_pixels = height * width

print(f"Full resolution: {width}x{height} = {total_pixels:,} pixels")

# ============================================================================
# STEP 2: ANALYZE COLOR DISTRIBUTION (DEBUG)
# ============================================================================
print("\nAnalyzing pixel colors...")

# Sample pixels to understand color distribution
sample_pixels = img_array[::10, ::10].reshape(-1, 3)
print(f"Sampled {len(sample_pixels)} pixels for analysis")
print(f"R range: {sample_pixels[:, 0].min()}-{sample_pixels[:, 0].max()}")
print(f"G range: {sample_pixels[:, 1].min()}-{sample_pixels[:, 1].max()}")
print(f"B range: {sample_pixels[:, 2].min()}-{sample_pixels[:, 2].max()}")

# ============================================================================
# STEP 3: IMPROVED COLOR DETECTION FOR SATELLITE
# ============================================================================
print("\nProcessing pixels with satellite-specific detection...")

track_map = []
track_positions = []  # Store all valid track positions

for row in range(height):
    track_row = []
    for col in range(width):
        r, g, b = img_array[row, col]
        
        # SATELLITE IMAGE COLOR DETECTION - REFINED
        
        # Dark blue/gray = Track asphalt (the actual racing surface)
        # Based on satellite imagery, track is darker with blue/gray tint
        if (50 <= r <= 130 and 60 <= g <= 140 and 70 <= b <= 150):
            cell = 'T'
            track_positions.append((row, col))
        
        # Tan/beige = Runoff areas (also drivable)
        elif (140 <= r <= 220 and 130 <= g <= 200 and 90 <= b <= 160):
            cell = 'T'
            track_positions.append((row, col))
        
        # Medium gray = Track borders (drivable)
        elif (80 <= r <= 160 and 90 <= g <= 170 and 100 <= b <= 180):
            cell = 'T'
            track_positions.append((row, col))
        
        # Green vegetation = Out of bounds
        elif g > r + 20 and g > b + 10 and g > 80:
            cell = '#'
        
        # Very dark = Shadows/barriers
        elif r < 50 and g < 50 and b < 50:
            cell = '#'
        
        # Very bright = Buildings/non-track
        elif r > 200 and g > 200 and b > 200:
            cell = '#'
        
        # Brown/dirt = Out of bounds
        elif r > 140 and g > 100 and b < 100 and r > g + 20:
            cell = '#'
        
        # Default to track for medium tones
        elif 60 <= r <= 200 and 70 <= g <= 200 and 80 <= b <= 200:
            cell = 'T'
            track_positions.append((row, col))
        
        else:
            cell = '#'
        
        track_row.append(cell)
    
    if (row + 1) % 100 == 0:
        print(f"  Processed {row + 1}/{height} rows...")
    
    track_map.append(track_row)

# Calculate statistics
track_cells = len(track_positions)
wall_cells = total_pixels - track_cells
track_percentage = (track_cells / total_pixels) * 100

print(f"\nDETECTION RESULTS:")
print(f"Track cells: {track_cells:,} ({track_percentage:.1f}%)")
print(f"Wall/OOB cells: {wall_cells:,} ({100-track_percentage:.1f}%)")

if track_cells == 0:
    print("\nERROR: No track detected! Color detection failed.")
    print("Showing sample of what was detected:")
    for row in range(min(20, height)):
        print("".join(track_map[row][:min(50, width)]))
    exit(1)

# ============================================================================
# STEP 4: RANDOM START POSITION ON TRACK
# ============================================================================
print("\nSelecting random start position...")

# Pick random track position
start_idx = np.random.randint(0, len(track_positions))
start_row, start_col = track_positions[start_idx]

# Mark as start
track_map[start_row][start_col] = 'S'

# Pick finish line (different random position, far from start)
finish_idx = np.random.randint(0, len(track_positions))
while abs(track_positions[finish_idx][0] - start_row) < height // 4:
    finish_idx = np.random.randint(0, len(track_positions))

finish_row, finish_col = track_positions[finish_idx]
track_map[finish_row][finish_col] = 'F'

print(f"Start: ({start_row}, {start_col})")
print(f"Finish: ({finish_row}, {finish_col})")

# Verify start is not on wall
if track_map[start_row][start_col] not in ['S', 'T', 'F']:
    print("ERROR: Start position is on wall!")
    exit(1)

# Show sample
print("\nMAP SAMPLE (around start position):")
sample_row_start = max(0, start_row - 100)
sample_row_end = min(height, start_row + 100)
sample_col_start = max(0, start_col - 100)
sample_col_end = min(width, start_col + 100)

for row in range(sample_row_start, sample_row_end):
    print("".join(track_map[row][sample_col_start:sample_col_end]))

# ============================================================================
# STEP 5: ENVIRONMENT SETUP
# ============================================================================
num_rows = height
num_cols = width

SLOW = 0
MEDIUM = 1
FAST = 2
num_velocity_levels = 3

state_space_size = num_rows * num_cols * num_velocity_levels

ACCELERATE = 0
COAST = 1
BRAKE = 2
TURN_LEFT = 3
TURN_RIGHT = 4
action_space_size = 5

print(f"\nENVIRONMENT:")
print(f"State space: {state_space_size:,} states")
print(f"Q-table size: {state_space_size * action_space_size * 4 / 1_000_000:.1f} MB")

# ============================================================================
# STEP 6: STATE ENCODING
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
# STEP 7: ENVIRONMENT DYNAMICS
# ============================================================================
def step(current_state, action, laps_completed):
    row, col, velocity = decode_state(current_state)
    
    new_velocity = velocity
    if action == ACCELERATE:
        new_velocity = min(FAST, velocity + 1)
    elif action == BRAKE:
        new_velocity = max(SLOW, velocity - 1)
    
    # Movement based on action
    if action in [ACCELERATE, COAST, BRAKE]:
        row_change = -1 if new_velocity == SLOW else -2
        col_change = 0
    elif action == TURN_LEFT:
        row_change = -1 if new_velocity == SLOW else -2
        col_change = -1
    elif action == TURN_RIGHT:
        row_change = -1 if new_velocity == SLOW else -2
        col_change = 1
    
    new_row = row + row_change
    new_col = col + col_change
    
    # Boundary check
    if new_row < 0 or new_row >= num_rows or new_col < 0 or new_col >= num_cols:
        reward = -100
        done = True
        new_row = start_row
        new_col = start_col
        new_velocity = SLOW
        laps_completed = 0
    else:
        cell_type = track_map[new_row][new_col]
        done = False
        
        if cell_type == '#':
            # Hit wall
            reward = -100
            done = True
            new_row = start_row
            new_col = start_col
            new_velocity = SLOW
            laps_completed = 0
        
        elif cell_type == 'F':
            # Crossed finish
            reward = 1000
            laps_completed += 1
            done = False
        
        elif cell_type in ['T', 'S']:
            # On track
            cells_moved = abs(row_change) + abs(col_change)
            reward = cells_moved * 1.0
        
        else:
            # Off track
            cells_moved = abs(row_change) + abs(col_change)
            reward = cells_moved * -5.0
    
    next_state = encode_state(new_row, new_col, new_velocity)
    return next_state, reward, done, laps_completed

def reset_environment():
    return encode_state(start_row, start_col, SLOW)

# ============================================================================
# STEP 8: Q-TABLE
# ============================================================================
print("\nInitializing Q-table...")
q_table = np.zeros((state_space_size, action_space_size), dtype=np.float32)
print(f"Memory: {q_table.nbytes / 1_000_000:.1f} MB")

# ============================================================================
# STEP 9: HYPERPARAMETERS
# ============================================================================
alpha = 0.01
gamma = 0.95
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995
num_episodes = 1000000
max_steps_per_episode = 10000

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

# ============================================================================
# STEP 10: TRAINING
# ============================================================================
rewards_per_episode = []
laps_per_episode = []
epsilon_values = []

epsilon = epsilon_start
start_time = time.time()

for episode in range(num_episodes):
    state = reset_environment()
    total_reward = 0
    laps_completed = 0
    done = False
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        if np.random.random() < epsilon:
            action = np.random.randint(0, action_space_size)
        else:
            action = np.argmax(q_table[state, :])
        
        next_state, reward, done, laps_completed = step(state, action, laps_completed)
        
        current_q = q_table[state, action]
        max_future_q = np.max(q_table[next_state, :])
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        q_table[state, action] = new_q
        
        state = next_state
        total_reward += reward
        steps += 1
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    rewards_per_episode.append(total_reward)
    laps_per_episode.append(laps_completed)
    epsilon_values.append(epsilon)
    
    if (episode + 1) % 500 == 0:
        elapsed = time.time() - start_time
        avg_reward = np.mean(rewards_per_episode[-100:])
        avg_laps = np.mean(laps_per_episode[-100:])
        crashes = sum(1 for r in rewards_per_episode[-100:] if r <= -50)
        
        print(f"Ep {episode + 1}/{num_episodes} | "
              f"Reward: {avg_reward:.1f} | "
              f"Laps: {avg_laps:.2f} | "
              f"Crashes: {crashes}/100 | "
              f"ε: {epsilon:.3f} | "
              f"{elapsed:.0f}s")

print(f"\nTraining complete: {(time.time()-start_time)/60:.1f} min")

# ============================================================================
# STEP 11: EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

test_episodes = 100
test_rewards = []
test_laps = []
test_crashes = 0

for episode in range(test_episodes):
    state = reset_environment()
    done = False
    total_reward = 0
    laps_completed = 0
    steps = 0
    
    while not done and steps < max_steps_per_episode:
        action = np.argmax(q_table[state, :])
        next_state, reward, done, laps_completed = step(state, action, laps_completed)
        state = next_state
        total_reward += reward
        steps += 1
    
    test_rewards.append(total_reward)
    test_laps.append(laps_completed)
    if total_reward <= -50:
        test_crashes += 1

print(f"\nAverage reward: {np.mean(test_rewards):.1f}")
print(f"Average laps: {np.mean(test_laps):.2f}")
print(f"Max laps: {max(test_laps)}")
print(f"Crashes: {test_crashes}/100")
print(f"Success: {(100-test_crashes)/100*100:.1f}%")

# ============================================================================
# STEP 12: PLOTS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

window = 100
smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
axes[0, 0].plot(smoothed_rewards, color='blue', linewidth=2)
axes[0, 0].set_title('Training Rewards')
axes[0, 0].grid(True, alpha=0.3)

smoothed_laps = np.convolve(laps_per_episode, np.ones(window)/window, mode='valid')
axes[0, 1].plot(smoothed_laps, color='green', linewidth=2)
axes[0, 1].set_title('Laps Completed')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epsilon_values, color='orange', linewidth=2)
axes[1, 0].set_title('Epsilon Decay')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].imshow(img_array)
axes[1, 1].plot([start_col], [start_row], 'go', markersize=10, label='Start')
axes[1, 1].plot([finish_col], [finish_row], 'ro', markersize=10, label='Finish')
axes[1, 1].set_title('Track Map')
axes[1, 1].legend()
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('laguna_seca_results.png', dpi=150)
print("\nSaved: laguna_seca_results.png")
plt.show()

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)