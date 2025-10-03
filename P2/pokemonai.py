import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyboy import PyBoy
import random
import os
import time
import pickle
import multiprocessing as mp
from multiprocessing import Manager
import signal
import sys

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_AGENTS = 5  # Bigger swarm
NUM_VISIBLE_AGENTS = 1  # Only show 1 emulator window
EPISODES_PER_AGENT = 150
MAX_STEPS_PER_EPISODE = 5000  # Much longer episodes for actual progress

LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0.99

WINDOW_POSITIONS = [
    (0, 0),  # Only first agent gets a visible window
]

# ============================================================================
# COMPLETELY NEW APPROACH: USE SCREEN PIXELS INSTEAD OF MEMORY
# ============================================================================

def get_screen_hash(pyboy):
    """Get hash of screen to detect changes."""
    screen = pyboy.screen.ndarray
    return hash(screen.tobytes())

def is_on_title_screen(pyboy):
    """Detect if we're on title screen by checking for Pokemon logo."""
    screen = pyboy.screen.ndarray
    # Check if there's a lot of white in upper portion (Pokemon logo)
    upper_third = screen[:60, :, :]
    white_pixels = np.sum(upper_third > 200)
    return white_pixels > 2000  # Title screen has lots of white

def create_playable_save(rom_path, save_path='playable_state.state'):
    """
    Create save by PLAYING through intro, not memory hacking.
    """
    print("\n" + "="*80)
    print("CREATING SAVE STATE BY PLAYING THROUGH INTRO")
    print("="*80)
    
    pyboy = PyBoy(rom_path, window="SDL2")
    pyboy.set_emulation_speed(0)
    
    # Boot
    print("\n1. Booting ROM...")
    for _ in range(200):
        pyboy.tick()
    
    # Get past title screen
    print("\n2. Getting past title screen...")
    attempts = 0
    while is_on_title_screen(pyboy) and attempts < 100:
        if attempts % 10 == 0:
            print(f"   Attempt {attempts}/100 - pressing START...")
        pyboy.button_press('start')
        for _ in range(20):
            pyboy.tick()
        pyboy.button_release('start')
        for _ in range(10):
            pyboy.tick()
        attempts += 1
    
    if is_on_title_screen(pyboy):
        print("   Still on title screen after 100 attempts!")
        print("   Trying A button...")
        for _ in range(50):
            pyboy.button_press('a')
            for _ in range(20):
                pyboy.tick()
            pyboy.button_release('a')
            for _ in range(10):
                pyboy.tick()
    
    # Skip through menus by spamming A
    print("\n3. Skipping menus...")
    for i in range(200):
        if i % 50 == 0:
            print(f"   Pressing A... {i}/200")
        pyboy.button_press('a')
        for _ in range(15):
            pyboy.tick()
        pyboy.button_release('a')
        for _ in range(5):
            pyboy.tick()
    
    # Test if we can move
    print("\n4. Testing game state...")
    initial_hash = get_screen_hash(pyboy)
    
    # Try moving
    for _ in range(5):
        pyboy.button_press('down')
        for _ in range(10):
            pyboy.tick()
        pyboy.button_release('down')
        for _ in range(5):
            pyboy.tick()
    
    after_hash = get_screen_hash(pyboy)
    screen_changed = initial_hash != after_hash
    
    print(f"   Screen changed: {screen_changed}")
    
    # Save state
    print(f"\n5. Saving to {save_path}...")
    with open(save_path, 'wb') as f:
        pyboy.save_state(f)
    
    print("✓ Save state created!")
    print("="*80 + "\n")
    
    # Keep window open briefly so user can see state
    print("Displaying state for 3 seconds...")
    pyboy.set_emulation_speed(1)
    time.sleep(3)
    
    pyboy.stop()
    return screen_changed

# ============================================================================
# SHARED CHECKPOINT MANAGEMENT
# ============================================================================

MASTER_CHECKPOINT_PATH = 'swarm_master_checkpoint.pth'
CHECKPOINT_LOCK = None

def init_checkpoint_lock(manager):
    """Initialize the shared checkpoint lock."""
    global CHECKPOINT_LOCK
    CHECKPOINT_LOCK = manager.Lock()

def save_master_checkpoint(agent_id, policy, optimizer, episode, episode_rewards, episode_screens, env_screen_visits):
    """Save to shared master checkpoint - all agents contribute to same file."""
    global CHECKPOINT_LOCK
    
    if CHECKPOINT_LOCK is None:
        return
    
    with CHECKPOINT_LOCK:
        # Load existing checkpoint if it exists
        if os.path.exists(MASTER_CHECKPOINT_PATH):
            try:
                master_checkpoint = torch.load(MASTER_CHECKPOINT_PATH)
            except:
                master_checkpoint = {'agents': {}}
        else:
            master_checkpoint = {'agents': {}}
        
        # Update this agent's data
        master_checkpoint['agents'][agent_id] = {
            'episode': episode,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'episode_screens': episode_screens,
            'screen_visits': env_screen_visits
        }
        
        # Save back to master file
        torch.save(master_checkpoint, MASTER_CHECKPOINT_PATH)

def load_master_checkpoint(agent_id, policy, optimizer):
    """Load this agent's data from master checkpoint."""
    if os.path.exists(MASTER_CHECKPOINT_PATH):
        try:
            master_checkpoint = torch.load(MASTER_CHECKPOINT_PATH)
            if agent_id in master_checkpoint.get('agents', {}):
                agent_data = master_checkpoint['agents'][agent_id]
                policy.load_state_dict(agent_data['policy_state_dict'])
                optimizer.load_state_dict(agent_data['optimizer_state_dict'])
                return (agent_data['episode'], 
                        agent_data['episode_rewards'], 
                        agent_data['episode_screens'],
                        agent_data.get('screen_visits', {}))
        except Exception as e:
            print(f"  Agent {agent_id}: Error loading checkpoint: {e}")
    return None

# Global flag for graceful shutdown
training_interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global training_interrupted
    print("\n\n" + "="*80)
    print("TRAINING INTERRUPTED - Saving checkpoints...")
    print("="*80)
    training_interrupted = True

# ============================================================================
# SIMPLE SCREEN-BASED POLICY NETWORK
# ============================================================================

class SimplePolicy(nn.Module):
    """Policy that works directly on screen pixels."""
    def __init__(self, action_size=8):  # 8 actions now (added 'start')
        super(SimplePolicy, self).__init__()
        
        # Convolutional layers for screen
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        # Calculate actual conv output size dynamically
        # This will work regardless of input dimensions
        self.conv_out_size = None
        
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.output_layer = None
    
    def _get_conv_output(self, shape):
        """Calculate conv output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.conv(dummy)
            return int(np.prod(output.size()))
    
    def forward(self, screen):
        # Initialize layers on first forward pass
        if self.conv_out_size is None:
            self.conv_out_size = self._get_conv_output(screen.shape[1:])
            # Add linear layer with correct input size
            self.fc.add_module('linear1', nn.Linear(self.conv_out_size, 128))
            self.output_layer = nn.Linear(128, 8)  # 8 actions
        
        x = self.conv(screen)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.output_layer(x)
        return torch.softmax(logits, dim=-1)

# ============================================================================
# SCREEN-BASED ENVIRONMENT
# ============================================================================

class ScreenBasedEnv:
    """Environment that uses screen changes instead of memory."""
    
    def __init__(self, rom_path, agent_id, window_pos, visible=False):
        self.agent_id = agent_id
        
        # Only show window for visible agents
        if visible:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_pos[0]},{window_pos[1]}"
            self.pyboy = PyBoy(rom_path, window="SDL2")
        else:
            self.pyboy = PyBoy(rom_path, window="null")
        
        self.pyboy.set_emulation_speed(0)
        
        # Load save
        if not os.path.exists('playable_state.state'):
            raise FileNotFoundError("playable_state.state not found!")
        
        with open('playable_state.state', 'rb') as f:
            self.pyboy.load_state(f)
        
        time.sleep(0.1)
        
        # Actions - added 'start' for menus
        self.actions = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'wait']
        
        # Tracking with visit counts
        self.screen_visits = {}  # hash -> visit count
        self.steps = 0
        self.episode_reward = 0
        self.total_episodes = 0
        self.best_reward = -float('inf')
        self.stuck_count = 0
        self.last_hash = None
        
        # Pokemon/Item detection via memory
        self.prev_party_count = 0
        self.prev_item_count = 0
        
        # Initialize
        initial_hash = get_screen_hash(self.pyboy)
        self.screen_visits[initial_hash] = 1
        self.last_hash = initial_hash
        
        if visible:
            print(f"  Agent {agent_id}: Initialized (VISIBLE window)")
        else:
            print(f"  Agent {agent_id}: Initialized (headless)")
    
    def get_screen_state(self):
        """Get screen as tensor."""
        screen = self.pyboy.screen.ndarray
        # Convert to grayscale
        gray = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
        # Normalize
        gray = gray / 255.0
        # Convert to tensor [1, 1, H, W]
        return torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0)
    
    def try_read_memory(self, address, default=0):
        """Safely try to read memory, return default if fails."""
        try:
            return self.pyboy.memory[address]
        except:
            return default
    
    def step(self, action_name):
        """Execute action with improved reward structure."""
        prev_hash = get_screen_hash(self.pyboy)
        
        # Try to detect Pokemon/Items before action
        prev_party = self.try_read_memory(0xD163, 0)  # Party count
        prev_item_count = sum(1 for i in range(0xD31E, 0xD345, 2) 
                             if self.try_read_memory(i, 0) != 0)
        
        # Execute action
        if action_name == 'wait':
            for _ in range(12):
                self.pyboy.tick()
        else:
            self.pyboy.button_press(action_name)
            for _ in range(10):
                self.pyboy.tick()
            self.pyboy.button_release(action_name)
            for _ in range(5):
                self.pyboy.tick()
        
        # Get new state
        new_hash = get_screen_hash(self.pyboy)
        
        # Check Pokemon/Items after action
        new_party = self.try_read_memory(0xD163, 0)
        new_item_count = sum(1 for i in range(0xD31E, 0xD345, 2) 
                            if self.try_read_memory(i, 0) != 0)
        
        # IMPROVED REWARD STRUCTURE
        reward = 0
        
        # 1. Screen change rewards (nuanced)
        if new_hash != prev_hash:
            self.stuck_count = 0
            
            if new_hash not in self.screen_visits:
                # NEW screen discovery - big reward
                reward += 10.0
                self.screen_visits[new_hash] = 1
                
                if len(self.screen_visits) % 50 == 0:
                    print(f"  Agent {self.agent_id}: {len(self.screen_visits)} unique screens!")
            else:
                # REVISITING a screen - diminishing returns
                visit_count = self.screen_visits[new_hash]
                
                if visit_count == 1:
                    reward += 2.0  # First revisit still good (might need to backtrack)
                elif visit_count == 2:
                    reward += 1.0  # Second revisit okay
                elif visit_count <= 5:
                    reward += 0.3  # A few more times is fine
                elif visit_count <= 10:
                    reward += 0.1  # Getting repetitive
                else:
                    reward -= 0.2  # Too much lingering in same area
                
                self.screen_visits[new_hash] += 1
        else:
            # No screen change - stuck
            self.stuck_count += 1
            if self.stuck_count > 5:
                reward -= 0.5  # Penalty for being stuck
        
        # 2. Pokemon rewards (always positive to gain, slight penalty to lose)
        if new_party > prev_party:
            reward += 50.0  # Caught a Pokemon!
            print(f"  Agent {self.agent_id}: Pokemon caught! Party: {prev_party} -> {new_party}")
        elif new_party < prev_party and prev_party > 0:
            reward -= 2.0  # Small penalty for losing Pokemon (depositing is sometimes needed)
        
        # 3. Item rewards (positive to gain, tiny penalty to discard)
        if new_item_count > prev_item_count:
            items_gained = new_item_count - prev_item_count
            reward += 5.0 * items_gained  # Items are valuable
            print(f"  Agent {self.agent_id}: Items gained! {prev_item_count} -> {new_item_count}")
        elif new_item_count < prev_item_count and prev_item_count > 0:
            reward -= 0.5  # Tiny penalty (sometimes need to toss items)
        
        # 4. Small time penalty (don't discourage long episodes)
        reward -= 0.03
        
        # Update tracking
        self.steps += 1
        self.episode_reward += reward
        self.last_hash = new_hash
        
        # Done if stuck for too long
        done = self.stuck_count > 100  # More lenient
        
        if done:
            print(f"  Agent {self.agent_id}: Episode ended (stuck {self.stuck_count} steps)")
        
        next_state = self.get_screen_state()
        
        return next_state, reward, done
    
    def reset(self):
        """Reset environment."""
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
        
        with open('playable_state.state', 'rb') as f:
            self.pyboy.load_state(f)
        
        time.sleep(0.05)
        
        # Reset tracking but KEEP screen_visits memory
        # This allows learning which screens lead to rewards
        self.steps = 0
        self.episode_reward = 0
        self.total_episodes += 1
        self.stuck_count = 0
        
        initial_hash = get_screen_hash(self.pyboy)
        self.last_hash = initial_hash
        
        return self.get_screen_state()
    
    def close(self):
        self.pyboy.stop()

# ============================================================================
# TRAINING
# ============================================================================

def train_agent(agent_id, rom_path, num_episodes, visible, global_stats, resume=False):
    """Train single agent with shared master checkpoint."""
    global training_interrupted
    
    if visible:
        print(f"\n{'='*60}")
        print(f"Agent {agent_id}: VISIBLE AGENT - Watch this one!")
        print(f"{'='*60}")
    else:
        print(f"\nAgent {agent_id}: Starting (headless)")
    
    # Only pass window position if visible
    window_pos = WINDOW_POSITIONS[0] if visible else (0, 0)
    env = ScreenBasedEnv(rom_path, agent_id, window_pos, visible=visible)
    policy = SimplePolicy(action_size=8)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Load from master checkpoint if resuming
    start_episode = 0
    episode_rewards = []
    episode_screens = []
    
    if resume:
        checkpoint_data = load_master_checkpoint(agent_id, policy, optimizer)
        if checkpoint_data:
            start_episode, episode_rewards, episode_screens, saved_visits = checkpoint_data
            # Restore screen visit history
            env.screen_visits = saved_visits
            print(f"  Agent {agent_id}: Resumed from episode {start_episode}")
        else:
            print(f"  Agent {agent_id}: No checkpoint found in master file, starting fresh")
    
    # Training loop
    for episode in range(start_episode, num_episodes):
        # Check if interrupted
        if training_interrupted:
            print(f"  Agent {agent_id}: Saving to master checkpoint at episode {episode}...")
            save_master_checkpoint(agent_id, policy, optimizer, episode, 
                                 episode_rewards, episode_screens, env.screen_visits)
            break
        
        # Storage
        states = []
        actions = []
        rewards = []
        
        # Reset
        state = env.reset()
        done = False
        steps = 0
        
        # Run episode
        while not done and steps < MAX_STEPS_PER_EPISODE:
            if training_interrupted:
                break
                
            # Get action
            with torch.no_grad():
                action_probs = policy(state)
            
            action_dist = torch.distributions.Categorical(action_probs)
            action_idx = action_dist.sample()
            action_name = env.actions[action_idx.item()]
            
            # Step
            next_state, reward, done = env.step(action_name)
            
            # Store
            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            
            state = next_state
            steps += 1
        
        if training_interrupted:
            break
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + DISCOUNT_FACTOR * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy
        policy_loss = 0
        for state, action, G in zip(states, actions, returns):
            action_probs = policy(state)
            action_dist = torch.distributions.Categorical(action_probs)
            log_prob = action_dist.log_prob(action)
            policy_loss -= log_prob * G
        
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        # Stats
        episode_rewards.append(env.episode_reward)
        episode_screens.append(len(env.screen_visits))
        
        # Update global stats
        global_stats[f'agent_{agent_id}'] = {
            'episode': episode,
            'reward': env.episode_reward,
            'best_reward': env.best_reward,
            'unique_screens': len(env.screen_visits),
            'steps': steps
        }
        
        # Save to master checkpoint every 5 episodes (to reduce file I/O contention)
        if episode % 5 == 0:
            save_master_checkpoint(agent_id, policy, optimizer, episode + 1,
                                 episode_rewards, episode_screens, env.screen_visits)
        
        # Print progress
        if episode % 10 == 0:
            avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"  Agent {agent_id} Episode {episode}: "
                  f"Avg={avg:.2f}, "
                  f"Screens={len(env.screen_visits)}, "
                  f"Best={env.best_reward:.2f}, "
                  f"Steps={steps}")
    
    # Final save to master checkpoint
    save_master_checkpoint(agent_id, policy, optimizer, episode + 1 if not training_interrupted else episode,
                         episode_rewards, episode_screens, env.screen_visits)
    
    # Also save individual final model for easy demo loading
    torch.save(policy.state_dict(), f'agent_{agent_id}_final.pth')
    with open(f'agent_{agent_id}_stats.pkl', 'wb') as f:
        pickle.dump({
            'rewards': episode_rewards,
            'screens': episode_screens
        }, f)
    
    print(f"\n✓ Agent {agent_id} complete!")
    print(f"  Total unique screens: {len(env.screen_visits)}")
    print(f"  Best episode reward: {env.best_reward:.2f}")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    
    env.close()

# ============================================================================
# PARALLEL TRAINING
# ============================================================================

def train_swarm(rom_path, num_agents=NUM_AGENTS, num_episodes=EPISODES_PER_AGENT, resume=False):
    """Train multiple agents - only 1 visible."""
    global training_interrupted
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n{'#'*80}")
    print(f"SWARM TRAINING: {num_agents} agents ({NUM_VISIBLE_AGENTS} visible, {num_agents - NUM_VISIBLE_AGENTS} headless)")
    if resume:
        print("RESUMING FROM MASTER CHECKPOINT")
    print(f"{'#'*80}")
    print("\nPress Ctrl+C at any time to stop and save progress")
    print(f"All agents save to: {MASTER_CHECKPOINT_PATH}")
    print()
    
    manager = Manager()
    global_stats = manager.dict()
    
    # Initialize shared checkpoint lock
    init_checkpoint_lock(manager)
    
    # Create processes - first one visible, rest headless
    processes = []
    for agent_id in range(num_agents):
        visible = (agent_id < NUM_VISIBLE_AGENTS)
        p = mp.Process(
            target=train_agent,
            args=(agent_id, rom_path, num_episodes, visible, global_stats, resume)
        )
        p.start()
        processes.append(p)
        time.sleep(0.3)  # Stagger starts
    
    # Wait
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nWaiting for agents to save to master checkpoint...")
        for p in processes:
            p.join(timeout=10)
    
    print(f"\n{'='*80}")
    if training_interrupted:
        print("TRAINING INTERRUPTED - Master checkpoint saved")
    else:
        print("TRAINING COMPLETE - Master checkpoint saved")
    print(f"{'='*80}")

# ============================================================================
# DEMO MODE
# ============================================================================

def demo_agent(agent_id, rom_path, max_steps=5000):
    """Watch a trained agent play."""
    print(f"\n{'='*80}")
    print(f"DEMO MODE - Agent {agent_id}")
    print(f"{'='*80}\n")
    
    # Check if model exists
    model_path = f'agent_{agent_id}_final.pth'
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found!")
        print(f"Train the agent first or choose a different agent.")
        return
    
    # Load policy
    policy = SimplePolicy(action_size=8)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    
    print(f"Loaded policy from {model_path}")
    
    # Create environment with visible window
    env = ScreenBasedEnv(rom_path, agent_id, WINDOW_POSITIONS[0], visible=True)
    env.pyboy.set_emulation_speed(1)  # Normal speed for watching
    
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print("\nWatching agent play...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while steps < max_steps:
            # Get action (no exploration)
            with torch.no_grad():
                action_probs = policy(state)
                action_idx = action_probs.argmax()
                action_name = env.actions[action_idx.item()]
            
            # Step
            next_state, reward, done = env.step(action_name)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                print(f"Steps: {steps}, Screens: {len(env.screen_visits)}, "
                      f"Reward: {total_reward:.2f}")
            
            if done:
                print(f"\nEpisode ended at step {steps}")
                # Reset and continue
                state = env.reset()
                total_reward = 0
            
            time.sleep(0.05)  # Slow down a bit for viewing
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    
    print(f"\nFinal stats:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Unique screens: {len(env.screen_visits)}")
    
    env.close()

# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(num_agents=NUM_AGENTS):
    """Comprehensive post-training analysis."""
    print("\n" + "="*80)
    print("POST-TRAINING ANALYSIS")
    print("="*80 + "\n")
    
    all_agent_data = []
    
    for agent_id in range(num_agents):
        try:
            with open(f'agent_{agent_id}_stats.pkl', 'rb') as f:
                data = pickle.load(f)
                rewards = data['rewards']
                screens = data.get('screens', [])
                
                all_agent_data.append({
                    'id': agent_id,
                    'rewards': rewards,
                    'screens': screens,
                    'total_episodes': len(rewards),
                    'avg_reward': np.mean(rewards),
                    'max_reward': np.max(rewards),
                    'final_10_avg': np.mean(rewards[-10:]),
                    'max_screens': max(screens) if screens else 0
                })
        except Exception as e:
            print(f"Agent {agent_id}: Could not load data ({e})")
    
    if not all_agent_data:
        print("No agent data found!\n")
        return
    
    # Sort by best performance
    all_agent_data.sort(key=lambda x: x['final_10_avg'], reverse=True)
    
    print("AGENT RANKINGS (by final 10 episode average):")
    print("-" * 80)
    for rank, agent in enumerate(all_agent_data, 1):
        print(f"{rank}. Agent {agent['id']}:")
        print(f"   Final 10 avg: {agent['final_10_avg']:.2f}")
        print(f"   Max reward: {agent['max_reward']:.2f}")
        print(f"   Average reward: {agent['avg_reward']:.2f}")
        print(f"   Max screens discovered: {agent['max_screens']}")
        print(f"   Total episodes: {agent['total_episodes']}")
        print()
    
    # Overall statistics
    print("="*80)
    print("SWARM STATISTICS:")
    print("-" * 80)
    
    all_final_avgs = [a['final_10_avg'] for a in all_agent_data]
    all_max_rewards = [a['max_reward'] for a in all_agent_data]
    all_max_screens = [a['max_screens'] for a in all_agent_data]
    
    print(f"Best performing agent: Agent {all_agent_data[0]['id']}")
    print(f"  Final 10 avg: {all_agent_data[0]['final_10_avg']:.2f}")
    print(f"  Max reward: {all_agent_data[0]['max_reward']:.2f}")
    print()
    
    print(f"Swarm averages:")
    print(f"  Average final 10 performance: {np.mean(all_final_avgs):.2f}")
    print(f"  Average max reward: {np.mean(all_max_rewards):.2f}")
    print(f"  Average max screens discovered: {np.mean(all_max_screens):.1f}")
    print()
    
    print(f"Swarm best:")
    print(f"  Best final 10 performance: {np.max(all_final_avgs):.2f}")
    print(f"  Best single episode: {np.max(all_max_rewards):.2f}")
    print(f"  Most screens discovered: {np.max(all_max_screens)}")
    print()
    
    # Learning progression for best agent
    best_agent = all_agent_data[0]
    print("="*80)
    print(f"BEST AGENT (Agent {best_agent['id']}) LEARNING PROGRESSION:")
    print("-" * 80)
    
    rewards = best_agent['rewards']
    milestones = [0, len(rewards)//4, len(rewards)//2, 3*len(rewards)//4, len(rewards)-1]
    
    for i, ep in enumerate(milestones):
        if ep < len(rewards):
            window_start = max(0, ep - 5)
            window_end = min(len(rewards), ep + 5)
            avg = np.mean(rewards[window_start:window_end])
            print(f"Episode {ep:3d}: {avg:8.2f} (±5 episode avg)")
    
    print("\n" + "="*80)
    print("Analysis complete! Check agent_X_final.pth files for trained policies.")
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    ROM_PATH = 'pokemon_red.gb'
    
    if not os.path.exists(ROM_PATH):
        print(f"ERROR: {ROM_PATH} not found!")
        exit(1)
    
    print("="*80)
    print("POKEMON SWARM TRAINER v4")
    print("="*80)
    
    # Create save state if needed
    if not os.path.exists('playable_state.state'):
        print("\nNo save state found. Creating one...")
        screen_worked = create_playable_save(ROM_PATH)
        if not screen_worked:
            print("\n⚠ WARNING: Screen didn't change during test!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                exit(1)
    
    # Main menu
    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("\n1. Train from scratch (start fresh training)")
        print("2. Continue training (resume from checkpoints)")
        print("3. Demo trained agent (watch agent play)")
        print("4. Analyze results (view training statistics)")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            # Train from scratch
            print("\n" + "="*80)
            print("TRAIN FROM SCRATCH")
            print("="*80)
            print(f"\nConfiguration:")
            print(f"  Total agents: {NUM_AGENTS}")
            print(f"  Visible agents: {NUM_VISIBLE_AGENTS}")
            print(f"  Episodes per agent: {EPISODES_PER_AGENT}")
            print(f"  Max steps per episode: {MAX_STEPS_PER_EPISODE}")
            print(f"\nReward structure:")
            print(f"  New screen: +10.0")
            print(f"  1st revisit: +2.0, 2nd: +1.0, 3-5th: +0.3, 6-10th: +0.1, 11+: -0.2")
            print(f"  Caught Pokemon: +50.0, Items: +5.0 each")
            print(f"  Actions: up, down, left, right, a, b, start, wait")
            
            confirm = input("\nStart training? (y/n): ").strip().lower()
            if confirm == 'y':
                train_swarm(ROM_PATH, resume=False)
                analyze_results()
        
        elif choice == '2':
            # Continue training
            print("\n" + "="*80)
            print("CONTINUE TRAINING")
            print("="*80)
            
            # Check if master checkpoint exists
            if not os.path.exists(MASTER_CHECKPOINT_PATH):
                print(f"\nNo master checkpoint found ({MASTER_CHECKPOINT_PATH})")
                print("Train from scratch first.")
                continue
            
            # Check how many agents have data in the checkpoint
            try:
                master_checkpoint = torch.load(MASTER_CHECKPOINT_PATH)
                agents_in_checkpoint = len(master_checkpoint.get('agents', {}))
                print(f"\nMaster checkpoint found with {agents_in_checkpoint} agent(s)")
                
                # Show some details
                for agent_id in sorted(master_checkpoint['agents'].keys())[:5]:
                    agent_data = master_checkpoint['agents'][agent_id]
                    print(f"  Agent {agent_id}: Episode {agent_data['episode']}, "
                          f"{len(agent_data['episode_rewards'])} episodes completed")
                
                if agents_in_checkpoint > 5:
                    print(f"  ... and {agents_in_checkpoint - 5} more agents")
                    
            except Exception as e:
                print(f"\nError reading checkpoint: {e}")
                continue
            
            confirm = input("\nResume training from master checkpoint? (y/n): ").strip().lower()
            if confirm == 'y':
                train_swarm(ROM_PATH, resume=True)
                analyze_results()
        
        elif choice == '3':
            # Demo mode
            print("\n" + "="*80)
            print("DEMO MODE")
            print("="*80)
            
            # Find available trained agents
            available_agents = [i for i in range(NUM_AGENTS) 
                              if os.path.exists(f'agent_{i}_final.pth')]
            
            if not available_agents:
                print("\nNo trained agents found! Train first.")
                continue
            
            print(f"\nAvailable trained agents: {available_agents}")
            agent_id = input(f"Enter agent ID to demo (default 0): ").strip()
            
            if agent_id == '':
                agent_id = 0
            else:
                try:
                    agent_id = int(agent_id)
                    if agent_id not in available_agents:
                        print(f"Agent {agent_id} not found!")
                        continue
                except ValueError:
                    print("Invalid agent ID!")
                    continue
            
            max_steps = input("Max steps (default 5000): ").strip()
            if max_steps == '':
                max_steps = 5000
            else:
                try:
                    max_steps = int(max_steps)
                except ValueError:
                    max_steps = 5000
            
            demo_agent(agent_id, ROM_PATH, max_steps)
        
        elif choice == '4':
            # Analyze results
            print("\n" + "="*80)
            print("ANALYZE RESULTS")
            print("="*80)
            analyze_results()
        
        elif choice == '5':
            # Exit
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice! Please enter 1-5.")