"""
LAGUNA SECA RACING - FEEDFORWARD DQN WITH GLOBAL VIEW

Uses simple feedforward network with expanded state representation:
- Agent's global position and velocity
- All checkpoint locations and distances
- Finish line location
- Much faster training than CNN approach
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAINING_IMAGE = 'img/laguna_seca.jpg'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("LAGUNA SECA FEEDFORWARD DQN")
print(f"Device: {DEVICE}")
print("=" * 70)

# ============================================================================
# LOAD AND PROCESS IMAGE
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

print(f"Resolution: {width}x{height} = {total_pixels:,} pixels")

# ============================================================================
# COLOR DETECTION
# ============================================================================
print("\nDetecting color-coded elements...")

track_map = []
track_positions = []
start_positions = []
finish_positions = []
checkpoint_positions = []

for row in range(height):
    track_row = []
    for col in range(width):
        r, g, b = img_array[row, col]
        
        if g > 200 and r < 50 and b < 50:
            cell = 'S'
            track_positions.append((row, col))
            start_positions.append((row, col))
        elif r > 200 and g < 50 and b < 50:
            cell = 'F'
            track_positions.append((row, col))
            finish_positions.append((row, col))
        elif b > 200 and r < 50 and g < 50:
            cell = 'C'
            track_positions.append((row, col))
            checkpoint_positions.append((row, col))
        elif (int(r) + int(g) + int(b)) / 3 < 100:
            cell = 'T'
            track_positions.append((row, col))
        elif (int(r) + int(g) + int(b)) / 3 > 200:
            cell = '#'
        else:
            cell = 'T'
            track_positions.append((row, col))
        
        track_row.append(cell)
    
    if (row + 1) % 100 == 0:
        print(f"  Processed {row + 1}/{height} rows...")
    
    track_map.append(track_row)

track_cells = len(track_positions)
print(f"\nDetection Results:")
print(f"Track: {track_cells:,} pixels ({track_cells/total_pixels*100:.1f}%)")
print(f"Start: {len(start_positions)} pixels")
print(f"Finish: {len(finish_positions)} pixels")
print(f"Checkpoints: {len(checkpoint_positions)} pixels")

# Set start/finish
start_row = int(np.mean([p[0] for p in start_positions]))
start_col = int(np.mean([p[1] for p in start_positions]))
finish_row = int(np.mean([p[0] for p in finish_positions]))
finish_col = int(np.mean([p[1] for p in finish_positions]))

print(f"Start: ({start_row}, {start_col})")
print(f"Finish: ({finish_row}, {finish_col})")

# Group checkpoints
checkpoint_regions = []
used = set()
for cp_row, cp_col in checkpoint_positions:
    if (cp_row, cp_col) in used:
        continue
    region = [(cp_row, cp_col)]
    used.add((cp_row, cp_col))
    for other_row, other_col in checkpoint_positions:
        if (other_row, other_col) in used:
            continue
        if np.sqrt((cp_row-other_row)**2 + (cp_col-other_col)**2) < 50:
            region.append((other_row, other_col))
            used.add((other_row, other_col))
    center = (int(np.mean([p[0] for p in region])), int(np.mean([p[1] for p in region])))
    checkpoint_regions.append({'center': center, 'positions': region, 'size': len(region)})

print(f"Checkpoint regions: {len(checkpoint_regions)}")
for i, region in enumerate(checkpoint_regions):
    print(f"  Region {i+1}: {region['center']}, {region['size']} pixels")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("INITIAL TRACK VISUALIZATION")
print("=" * 70)

map_colors = np.zeros((height, width, 3))
text_grid = []

for row in range(height):
    text_row = []
    for col in range(width):
        cell = track_map[row][col]
        if cell == 'T':
            map_colors[row, col] = [0.3, 0.3, 0.3]
            text_row.append('T')
        elif cell == '#':
            map_colors[row, col] = [0.95, 0.95, 0.95]
            text_row.append('#')
        elif cell == 'S':
            map_colors[row, col] = [0.0, 0.9, 0.0]
            text_row.append('S')
        elif cell == 'F':
            map_colors[row, col] = [0.9, 0.0, 0.0]
            text_row.append('F')
        elif cell == 'C':
            map_colors[row, col] = [0.0, 0.0, 0.9]
            text_row.append('C')
    text_grid.append(text_row)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(map_colors)
axes[0].set_title('Detected Track (Color-Coded)', fontsize=14, fontweight='bold')
axes[0].axis('off')

text_img = map_colors.copy()
axes[1].imshow(text_img)
axes[1].set_title('Grid Representation (T=Track, #=Wall, S=Start, F=Finish, C=Checkpoint)', 
                  fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('track_initial_detection.png', dpi=150, bbox_inches='tight')
print("Saved: track_initial_detection.png")
plt.show()

# ============================================================================
# FEEDFORWARD DQN NETWORK
# ============================================================================
class FeedForwardDQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(FeedForwardDQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# REPLAY BUFFER
# ============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# ENVIRONMENT WITH GLOBAL STATE
# ============================================================================
class RacingEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.row = start_row
        self.col = start_col
        self.velocity = 0.0
        self.visited_checkpoints = set()
        self.total_steps = 0
        return self.get_state()
    
    def get_state(self):
        """
        Global state representation with full visibility:
        - Normalized position (x, y)
        - Normalized velocity
        - Distance to each checkpoint (normalized)
        - Which checkpoints are visited (binary flags)
        - Distance to finish (normalized)
        - Direction vector to nearest unvisited checkpoint
        """
        state = []
        
        # Agent position (normalized to 0-1)
        state.append(self.row / height)
        state.append(self.col / width)
        
        # Velocity (normalized to 0-1)
        state.append(self.velocity / 100.0)
        
        # Distance and direction to each checkpoint
        for i, region in enumerate(checkpoint_regions):
            cp_row, cp_col = region['center']
            
            # Distance (normalized)
            distance = np.sqrt((self.row - cp_row)**2 + (self.col - cp_col)**2)
            norm_distance = distance / np.sqrt(height**2 + width**2)
            state.append(norm_distance)
            
            # Direction vector (normalized)
            if distance > 0:
                state.append((cp_row - self.row) / distance)
                state.append((cp_col - self.col) / distance)
            else:
                state.append(0.0)
                state.append(0.0)
            
            # Visited flag
            state.append(1.0 if i in self.visited_checkpoints else 0.0)
        
        # Distance and direction to finish
        finish_distance = np.sqrt((self.row - finish_row)**2 + (self.col - finish_col)**2)
        norm_finish_distance = finish_distance / np.sqrt(height**2 + width**2)
        state.append(norm_finish_distance)
        
        if finish_distance > 0:
            state.append((finish_row - self.row) / finish_distance)
            state.append((finish_col - self.col) / finish_distance)
        else:
            state.append(0.0)
            state.append(0.0)
        
        return torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
    
    def step(self, action):
        # Action mapping: 0=Accel, 1=Coast, 2=Brake, 3=TurnLeft, 4=TurnRight
        if action == 0:
            self.velocity = min(100.0, self.velocity + 15.0)
            row_change, col_change = -2, 0
        elif action == 1:
            self.velocity = max(0.0, self.velocity - 2.0)
            row_change, col_change = -1, 0
        elif action == 2:
            self.velocity = max(0.0, self.velocity - 25.0)
            row_change, col_change = -1, 0
        elif action == 3:
            row_change, col_change = -1, -1
        elif action == 4:
            row_change, col_change = -1, 1
        
        prev_row, prev_col = self.row, self.col
        new_row = self.row + row_change
        new_col = self.col + col_change
        self.total_steps += 1
        
        # Bounds check
        if new_row < 0 or new_row >= height or new_col < 0 or new_col >= width:
            reward = -100.0
            done = True
            self.reset()
            return self.get_state(), reward, done, {'crash': True}
        
        cell = track_map[new_row][new_col]
        done = False
        info = {}
        
        # SPARSE REWARD STRUCTURE - Only reward achievements
        
        if cell == '#':
            # Wall crash
            reward = -100.0
            done = True
            self.reset()
            return self.get_state(), reward, done, {'crash': True}
        
        elif cell == 'F':
            # Finish line
            min_checkpoints = max(1, len(checkpoint_regions) - 1)
            if len(self.visited_checkpoints) < min_checkpoints:
                # Finish without checkpoints - massive penalty
                reward = -100.0
                done = True
                info['finish_blocked'] = True
                if np.random.random() < 0.02:
                    print(f"  Finish blocked: {len(self.visited_checkpoints)}/{min_checkpoints} CP")
                self.reset()
            else:
                # SUCCESS - huge reward
                reward = 50.0 + len(self.visited_checkpoints) * 500.0
                done = True
                info['success'] = True
                print(f"  *** SUCCESS! Steps: {self.total_steps}, CP: {len(self.visited_checkpoints)} ***")
                self.reset()
        
        elif cell == 'C':
            # Checkpoint collection
            current_cp = None
            for i, region in enumerate(checkpoint_regions):
                for cp_row, cp_col in region['positions']:
                    if new_row == cp_row and new_col == cp_col:
                        current_cp = i
                        break
            
            if current_cp is not None and current_cp not in self.visited_checkpoints:
                # First time collecting this checkpoint - HUGE reward
                reward = 2000.0
                self.visited_checkpoints.add(current_cp)
                info['checkpoint'] = current_cp
                print(f"  ✓ Checkpoint {current_cp+1}/{len(checkpoint_regions)} collected!")
            else:
                # Already visited or not a checkpoint - zero reward
                reward = 0.0
        
        else:
            # Normal track movement - ZERO reward
            # Agent must find checkpoints through exploration, not gradient following
            reward = 0.0
        
        self.row = new_row
        self.col = new_col
        
        return self.get_state(), reward, done, info

# ============================================================================
# TRAINING SETUP
# ============================================================================
print("\n" + "=" * 70)
print("FEEDFORWARD DQN SETUP")
print("=" * 70)

env = RacingEnv()
num_actions = 5

# Calculate state size
# Position(2) + Velocity(1) + Per_Checkpoint[Distance(1) + Direction(2) + Visited(1)] + Finish[Distance(1) + Direction(2)]
state_size = 3 + len(checkpoint_regions) * 4 + 3
print(f"State size: {state_size} features")

policy_net = FeedForwardDQN(state_size, num_actions).to(DEVICE)
target_net = FeedForwardDQN(state_size, num_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(50000)

# Hyperparameters
batch_size = 128
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.9999  # Much slower decay - maintains exploration
target_update = 100
num_episodes = 10000  # More episodes for sparse rewards

print(f"Network parameters: {sum(p.numel() for p in policy_net.parameters()):,}")
print(f"Replay buffer: 50,000")
print(f"Episodes: {num_episodes}")

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)
print("SPARSE REWARD STRUCTURE:")
print("- Normal track movement: 0 reward")
print("- Checkpoint collection: +2000")
print("- Finish (with checkpoints): +5000 + 500 per checkpoint")
print("- Crash/OOB: -100")
print("- Finish blocked: -1000")
print("\nAgent MUST explore to find rewards (high epsilon maintained)")
print("Starting...\n")

epsilon = epsilon_start
rewards_per_episode = []
checkpoints_per_episode = []
epsilon_values = []

milestone_paths = {}
milestones = [100, 500, 1000, 2500, 4999]

start_time = time.time()

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    path = []
    
    while not done and steps < 2000:
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].item()
        
        path.append((env.row, env.col, env.velocity))
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        steps += 1
        
        # Train
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.cat(states)
            actions = torch.LongTensor(actions).unsqueeze(1).to(DEVICE)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
            next_states = torch.cat(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)
            
            current_q = policy_net(states).gather(1, actions)
            next_q = target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + gamma * next_q * (1 - dones)
            
            loss = nn.MSELoss()(current_q, target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if episode in milestones:
        milestone_paths[episode] = path.copy()
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    rewards_per_episode.append(total_reward)
    checkpoints_per_episode.append(len(env.visited_checkpoints))
    epsilon_values.append(epsilon)
    
    elapsed = time.time() - start_time
    avg_reward = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
    avg_checkpoints = np.mean(checkpoints_per_episode[-100:]) if len(checkpoints_per_episode) >= 100 else np.mean(checkpoints_per_episode)
    buffer_status = f"{len(replay_buffer)}/{replay_buffer.buffer.maxlen}"
    
    should_print = False
    if episode < 100 and (episode + 1) % 10 == 0:
        should_print = True
    elif episode < 1000 and (episode + 1) % 50 == 0:
        should_print = True
    elif (episode + 1) % 200 == 0:
        should_print = True
    
    if should_print:
        learning_status = "WARMUP" if len(replay_buffer) < batch_size else "LEARNING"
        print(f"Ep {episode+1:>5}/{num_episodes} [{learning_status}] | "
              f"R: {avg_reward:>7.1f} | "
              f"CP: {avg_checkpoints:>4.2f}/{len(checkpoint_regions)} | "
              f"Steps: {steps:>4} | "
              f"ε: {epsilon:.3f} | "
              f"{elapsed:.0f}s")

print(f"\nTraining complete: {(time.time()-start_time)/60:.1f} min")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 70)
print("CREATING RESULTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

window = 100
if len(rewards_per_episode) > window:
    smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(smoothed_rewards, color='blue', linewidth=2)
axes[0, 0].set_title('Training Rewards', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].grid(True, alpha=0.3)

if len(checkpoints_per_episode) > window:
    smoothed_checkpoints = np.convolve(checkpoints_per_episode, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(smoothed_checkpoints, color='green', linewidth=2)
axes[0, 1].set_title('Checkpoints Collected', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Checkpoints')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epsilon_values, color='orange', linewidth=2)
axes[1, 0].set_title('Epsilon Decay', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Epsilon')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].imshow(map_colors, alpha=0.5)

colors = plt.cm.viridis(np.linspace(0, 1, len(milestone_paths)))
for idx, (ep, path) in enumerate(milestone_paths.items()):
    if len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        axes[1, 1].plot(cols, rows, color=colors[idx], alpha=0.8, linewidth=2.5,
                       label=f"Ep {ep}")

axes[1, 1].set_title('Learning Progress: Milestone Paths', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='upper right', fontsize=9)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('feedforward_dqn_results.png', dpi=150, bbox_inches='tight')
print("Saved: feedforward_dqn_results.png")
plt.show()

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)