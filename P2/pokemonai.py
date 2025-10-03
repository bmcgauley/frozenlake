import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyboy import PyBoy
from collections import deque
import random
import os
import json
import matplotlib.pyplot as plt

# Action space: Game Boy buttons (8 actions)
# 0: A, 1: B, 2: Start, 3: Select, 4: Up, 5: Down, 6: Left, 7: Right
ACTIONS = ['a', 'b', 'start', 'select', 'up', 'down', 'left', 'right']

class PokemonEnv:
    def __init__(self, rom_path, window="null"):
        self.pyboy = PyBoy(rom_path, window=window)
        self.pyboy.set_emulation_speed(0)  # Fast emulation
        
        # Skip the intro sequence properly
        print("Skipping intro sequence...")
        for _ in range(200):  # Wait for intro to load
            self.pyboy.tick()
        
        # Press start to get to menu
        self.pyboy.button_press('start')
        for _ in range(60):
            self.pyboy.tick()
        
        # Choose "New Game" (A button)
        self.pyboy.button_press('a')
        for _ in range(120):
            self.pyboy.tick()
        
        # Skip through the naming screens by pressing A repeatedly
        for _ in range(10):  # Press A multiple times to skip naming
            self.pyboy.button_press('a')
            for _ in range(30):
                self.pyboy.tick()
        
        # Wait for the game to actually start
        for _ in range(300):
            self.pyboy.tick()
        
        # Check if we have HP now
        current_hp = self.get_hp()
        print(f"After initialization - HP: {current_hp}")
        
        if current_hp == 0:
            print("Warning: HP is 0, manually setting up a basic game state")
            # Manually set up a basic playable state
            self.pyboy.memory[0xD16C] = 20  # HP
            self.pyboy.memory[0xD18B] = 5   # Level
            self.pyboy.memory[0xD163] = 1   # Party size (1 Pokemon)
            # Set some basic Pokemon data (simplified)
            self.pyboy.memory[0xD164] = 1   # First Pokemon species (Bulbasaur)
            self.pyboy.memory[0xD165] = 5   # Level
            self.pyboy.memory[0xD16D] = 20  # Max HP
            self.pyboy.memory[0xD16E] = 20  # Current HP
        
        self.initial_state_file = 'initial_state.state'
        with open(self.initial_state_file, 'wb') as f:
            self.pyboy.save_state(f)
        
        self.prev_level = 0
        self.prev_hp = 0
        self.prev_money = 0
        self.prev_badges = 0
        self.prev_event_flags = 0
        self.prev_seen_pokemon = 0
        self.prev_caught_pokemon = 0
        self.prev_party_size = 0
        self.prev_opponent_level = 0
        self.explored_coords = set()
        self.reset()

    def reset(self):
        with open(self.initial_state_file, 'rb') as f:
            self.pyboy.load_state(f)
        
        # Ensure we have HP
        current_hp = self.get_hp()
        if current_hp == 0:
            print("Warning: HP is 0 in reset, setting up basic game state")
            # Manually set up a basic playable state
            self.pyboy.memory[0xD16C] = 20  # HP
            self.pyboy.memory[0xD18B] = 5   # Level
            self.pyboy.memory[0xD163] = 1   # Party size (1 Pokemon)
            # Set some basic Pokemon data (simplified)
            self.pyboy.memory[0xD164] = 1   # First Pokemon species (Bulbasaur)
            self.pyboy.memory[0xD165] = 5   # Level
            self.pyboy.memory[0xD16D] = 20  # Max HP
            self.pyboy.memory[0xD16E] = 20  # Current HP
        
        self.prev_level = self.get_level()
        self.prev_hp = self.get_hp()
        self.prev_money = self.get_money()
        self.prev_badges = self.get_badges()
        self.prev_event_flags = self.get_event_flags()
        self.prev_seen_pokemon = self.get_seen_pokemon()
        self.prev_caught_pokemon = self.get_caught_pokemon()
        self.prev_party_size = self.get_party_size()
        self.prev_opponent_level = self.get_opponent_level()
        self.explored_coords = set([self.get_position()])
        
        # Debug: print initial state
        print(f"Reset - HP: {self.prev_hp}, Level: {self.prev_level}, Money: {self.prev_money}, Badges: {self.prev_badges}")
        
        return self.get_state()

    def step(self, action):
        # Perform action
        self.pyboy.button_press(action)
        self.pyboy.tick()
        self.pyboy.tick()  # Additional tick for stability

        # Get next state
        next_state = self.get_state()

        # Calculate reward
        reward = self.calculate_reward()

        # Update previous values
        self.prev_level = self.get_level()
        self.prev_hp = self.get_hp()
        self.prev_money = self.get_money()
        self.prev_badges = self.get_badges()
        self.prev_event_flags = self.get_event_flags()
        self.prev_seen_pokemon = self.get_seen_pokemon()
        self.prev_caught_pokemon = self.get_caught_pokemon()
        self.prev_party_size = self.get_party_size()
        self.prev_opponent_level = self.get_opponent_level()
        self.explored_coords.add(self.get_position())

        # Done if game over (HP = 0) or arbitrary episode length
        done = self.is_done()

        return next_state, reward, done, {}

    def get_state(self):
        # State: Downscaled screen (84x84 grayscale)
        screen = self.pyboy.screen.ndarray
        # Convert to grayscale
        screen_gray = np.dot(screen[...,:3], [0.2989, 0.5870, 0.1140])
        state = torch.tensor(screen_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        # Resize to 84x84
        state = nn.functional.interpolate(state, size=(84, 84), mode='bilinear', align_corners=False)
        return state

    def get_level(self):
        return self.pyboy.memory[0xD18B]

    def get_hp(self):
        return self.pyboy.memory[0xD16C]

    def get_money(self):
        # Money is 3 bytes, little endian
        return self.pyboy.memory[0xD347] + (self.pyboy.memory[0xD348] << 8) + (self.pyboy.memory[0xD349] << 16)

    def get_badges(self):
        return bin(self.pyboy.memory[0xD356]).count('1')

    def get_event_flags(self):
        # Count set event flags from 0xD747 to 0xD886
        return sum(bin(self.pyboy.memory[i]).count('1') for i in range(0xD747, 0xD886))

    def get_seen_pokemon(self):
        # Seen Pokemon flags from 0xD30A to 0xD31D
        return sum(bin(self.pyboy.memory[i]).count('1') for i in range(0xD30A, 0xD31D))

    def get_caught_pokemon(self):
        # Caught Pokemon flags from 0xD2F7 to 0xD30A
        return sum(bin(self.pyboy.memory[i]).count('1') for i in range(0xD2F7, 0xD30A))

    def get_party_size(self):
        return self.pyboy.memory[0xD163]

    def get_opponent_level(self):
        # Opponent Pokemon levels
        opp_levels = [self.pyboy.memory[addr] for addr in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]
        return max(opp_levels) if opp_levels else 0

    def get_position(self):
        x = self.pyboy.memory[0xD362]
        y = self.pyboy.memory[0xD361]
        map_n = self.pyboy.memory[0xD35E]
        return (x, y, map_n)

    def bit_count(self, bits):
        return bin(bits).count('1')

    def calculate_reward(self):
        reward = 0
        current_level = self.get_level()
        current_hp = self.get_hp()
        current_money = self.get_money()
        current_badges = self.get_badges()
        current_event_flags = self.get_event_flags()
        current_seen_pokemon = self.get_seen_pokemon()
        current_caught_pokemon = self.get_caught_pokemon()
        current_party_size = self.get_party_size()
        current_opponent_level = self.get_opponent_level()
        current_position = self.get_position()

        # Reward for leveling up Pokemon
        if current_level > self.prev_level:
            reward += (current_level - self.prev_level) * 5

        # Penalty for losing HP (but not too harsh)
        if current_hp < self.prev_hp:
            reward -= (self.prev_hp - current_hp) * 0.1

        # Reward for gaining money (winning battles)
        if current_money > self.prev_money:
            reward += (current_money - self.prev_money) // 100  # Scale down

        # Reward for earning badges (beating gyms)
        if current_badges > self.prev_badges:
            reward += (current_badges - self.prev_badges) * 50

        # Reward for triggering new events
        if current_event_flags > self.prev_event_flags:
            reward += (current_event_flags - self.prev_event_flags) * 2

        # Reward for seeing new Pokemon
        if current_seen_pokemon > self.prev_seen_pokemon:
            reward += (current_seen_pokemon - self.prev_seen_pokemon) * 10

        # Reward for catching new Pokemon (unseen before)
        if current_caught_pokemon > self.prev_caught_pokemon:
            reward += (current_caught_pokemon - self.prev_caught_pokemon) * 20

        # Reward for increasing party size (catching Pokemon)
        if current_party_size > self.prev_party_size:
            reward += (current_party_size - self.prev_party_size) * 15

        # Reward for exploring new tiles
        if current_position not in self.explored_coords:
            reward += 1  # Small reward for new exploration

        # Penalty for dying (HP = 0)
        if current_hp == 0:
            reward -= 100

        # Small penalty per step to encourage efficiency
        reward -= 0.01

        # Debug prints (only when reward changes significantly)
        if abs(reward) > 0.1:
            print(f"Reward: {reward:.2f}, Level: {current_level}, HP: {current_hp}, Money: {current_money}, Badges: {current_badges}, Seen: {current_seen_pokemon}, Caught: {current_caught_pokemon}, Events: {current_event_flags}, Party: {current_party_size}")

        return reward

    def is_done(self):
        # Done if HP is 0 (died) or all 8 badges earned or too many steps
        return self.get_hp() == 0 or self.get_badges() >= 8

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_out(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

# Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Training function
def train_dqn(model, target_model, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states)
    next_states = torch.cat(next_states)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = target_model(next_states).max(1)[0]
    targets = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Main training loop
def train_agent(rom_path, num_episodes=1000, max_steps=10000, batch_size=32, gamma=0.99,
                epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10,
                save_path='pokemon_dqn.pth', stats_path='training_stats.json'):
    env = PokemonEnv(rom_path)
    model = DQN((1, 84, 84), len(ACTIONS))
    target_model = DQN((1, 84, 84), len(ACTIONS))
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(10000)

    epsilon = epsilon_start
    episode_stats = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        episode_badges = 0
        episode_levels = 0
        episode_exploration = 0

        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                with torch.no_grad():
                    action_idx = model(state).argmax().item()
                    action = ACTIONS[action_idx]

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, ACTIONS.index(action), reward, next_state, done)

            train_dqn(model, target_model, optimizer, replay_buffer, batch_size, gamma)

            state = next_state
            total_reward += reward
            steps += 1

        # Collect episode stats
        final_badges = env.get_badges()
        final_level = env.get_level()
        final_exploration = len(env.explored_coords)
        
        episode_stats.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'steps': steps,
            'badges': final_badges,
            'level': final_level,
            'exploration': final_exploration,
            'epsilon': epsilon
        })

        # Update target network
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Steps: {steps}, Badges: {final_badges}, Level: {final_level}, Exploration: {final_exploration}, Epsilon: {epsilon:.3f}")

        # Save model periodically
        if (episode + 1) % 100 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # Save final stats
    import json
    with open(stats_path, 'w') as f:
        json.dump(episode_stats, f, indent=2)
    print(f"Training stats saved to {stats_path}")

    # Final save
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

    return episode_stats

# Function to play with trained model
def play_agent(rom_path, model_path='pokemon_dqn.pth', max_steps=1000, window="SDL2"):
    env = PokemonEnv(rom_path, window=window)
    if window == "SDL2":
        env.pyboy.set_emulation_speed(1)  # Normal speed for watching
    model = DQN((1, 84, 84), len(ACTIONS))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < max_steps:
        with torch.no_grad():
            action_idx = model(state).argmax().item()
            action = ACTIONS[action_idx]

        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        steps += 1

        print(f"Step {steps}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

    print(f"Play complete. Total Reward: {total_reward:.2f}")

# Function to create visualizations from training stats
def create_visualizations(stats_path='training_stats.json'):
    if not os.path.exists(stats_path):
        print(f"Stats file {stats_path} not found.")
        return
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    episodes = [s['episode'] for s in stats]
    rewards = [s['total_reward'] for s in stats]
    badges = [s['badges'] for s in stats]
    levels = [s['level'] for s in stats]
    explorations = [s['exploration'] for s in stats]
    epsilons = [s['epsilon'] for s in stats]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pokemon AI Training Results')
    
    # Rewards over time
    axes[0, 0].plot(episodes, rewards)
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Badges over time
    axes[0, 1].plot(episodes, badges, 'r-o')
    axes[0, 1].set_title('Badges Earned')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Badges')
    axes[0, 1].grid(True)
    
    # Levels over time
    axes[0, 2].plot(episodes, levels, 'g-s')
    axes[0, 2].set_title('Pokemon Levels')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Level')
    axes[0, 2].grid(True)
    
    # Exploration over time
    axes[1, 0].plot(episodes, explorations, 'm-^')
    axes[1, 0].set_title('Tiles Explored')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Tiles')
    axes[1, 0].grid(True)
    
    # Epsilon over time
    axes[1, 1].plot(episodes, epsilons, 'c-d')
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True)
    
    # Cumulative rewards
    cum_rewards = np.cumsum(rewards)
    axes[1, 2].plot(episodes, cum_rewards, 'b-*')
    axes[1, 2].set_title('Cumulative Reward')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Cumulative Reward')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('pokemon_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved to pokemon_training_results.png")

if __name__ == "__main__":
    rom_path = 'pokemon_red.gb'  # Update this to your ROM path
    if not os.path.exists(rom_path):
        print(f"ROM file {rom_path} not found. Please place the Pokemon ROM in the project directory.")
    else:
        # Train the agent
        print("Starting training...")
        stats = train_agent(rom_path, num_episodes=100, max_steps=1000)  # Short episodes for testing
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations()
        
        # Play with trained model (set window="null" for headless, "SDL2" to watch)
        print("Playing trained model...")
        play_agent(rom_path, window="SDL2")
