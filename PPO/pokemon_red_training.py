"""
Universal Game Boy Game Learner
PPO-based agent that learns to play ANY Game Boy game through visual exploration

This implementation creates a generalized game-playing agent that works across
different Game Boy titles by focusing on visual novelty and exploration rather
than game-specific mechanics.

PHILOSOPHY:
- Reward visual novelty (seeing new screens) 
- Minimal time penalty to encourage action
- No game-specific knowledge or rewards
- Universal approach that works for Pokemon, Mario, Zelda, etc.

Requirements:
- Game Boy ROM file (any .gb file)
- PyBoy emulator for Game Boy emulation
- Stable-baselines3 for PPO algorithm
- PyTorch as backend
- Gymnasium for environment interface

Architecture:
- Custom Gymnasium environment wrapping PyBoy emulator
- Screen-based novelty detection for exploration rewards
- Minimal reward structure for universal game learning
- Parallel environment training for speed
- Real-time training metrics and visualization
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import time
from datetime import datetime
import json
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Import PyBoy for Game Boy emulation
from pyboy import PyBoy

# Import stable-baselines3 for PPO algorithm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Import preprocessing utilities
from skimage.transform import resize

# Import custom screen state tracker
from screen_state_tracker import ScreenStateTracker

print("=" * 80)
print("UNIVERSAL GAME BOY GAME LEARNER - TRAINING SYSTEM")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# ============================================================================
# MEMORY ADDRESSES - Pokemon Red RAM Locations
# These addresses are from the Pokemon Red disassembly (PRET project)
# ============================================================================

# Player position and map
PLAYER_X_ADDRESS = 0xD362
PLAYER_Y_ADDRESS = 0xD361
MAP_ID_ADDRESS = 0xD35E

# Party Pokemon data
PARTY_SIZE_ADDRESS = 0xD163
PARTY_LEVEL_START = 0xD18C  # Array of party Pokemon levels
PARTY_HP_START = 0xD16C  # Array of current HP (2 bytes each)
PARTY_MAX_HP_START = 0xD18C  # Array of max HP (2 bytes each)

# Badge flags (8 badges = 8 bits)
BADGE_FLAGS_ADDRESS = 0xD356

# Event flags for story progression
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD761  # Total of ~320 event flags

# Battle data
ENEMY_MON_LEVEL_ADDRESS = 0xD127
BATTLE_TYPE_ADDRESS = 0xD057  # 0 = no battle, others = battle type
IN_BATTLE_ADDRESS = 0xD057

# Menu tracking
MENU_STATE_ADDRESS = 0xD0E0  # Current menu state
JOYPAD_STATE_ADDRESS = 0xD0E1  # Last joypad input

# Pokemon seen/owned
POKEDEX_OWNED_START = 0xD2F7  # Bitfield of owned Pokemon
POKEDEX_SEEN_START = 0xD30A  # Bitfield of seen Pokemon

# Money (3 bytes, BCD format)
MONEY_ADDRESS_1 = 0xD347
MONEY_ADDRESS_2 = 0xD348
MONEY_ADDRESS_3 = 0xD349

# Elite Four progression
ELITE_FOUR_DEFEATS = 0xD847  # Custom tracking for Elite Four

# ============================================================================
# MILESTONE DEFINITIONS - Major Progress Points
# ============================================================================

MILESTONES = {
    'starter_obtained': 'Got starter Pokemon',
    'first_battle_won': 'Won first battle',
    'first_gym_badge': 'Defeated Brock (Boulder Badge)',
    'second_gym_badge': 'Defeated Misty (Cascade Badge)',
    'third_gym_badge': 'Defeated Lt. Surge (Thunder Badge)',
    'fourth_gym_badge': 'Defeated Erika (Rainbow Badge)',
    'fifth_gym_badge': 'Defeated Koga (Soul Badge)',
    'sixth_gym_badge': 'Defeated Sabrina (Marsh Badge)',
    'seventh_gym_badge': 'Defeated Blaine (Volcano Badge)',
    'eighth_gym_badge': 'Defeated Giovanni (Earth Badge)',
    'elite_four_entered': 'Entered Elite Four',
    'elite_four_defeated': 'Defeated Elite Four',
    'champion_defeated': 'Became Pokemon Champion!'
}

# ============================================================================
# CUSTOM GYMNASIUM ENVIRONMENT - Pokemon Red
# ============================================================================

class PokemonRedEnv(gym.Env):
    """
    Custom Gymnasium environment for Pokemon Red speedrunning.
    
    Observation Space: (120, 128, 3) RGB image
    Action Space: Discrete(9) - 4 directions, 2 buttons, start, select, no-op
    
    This environment wraps the PyBoy Game Boy emulator and implements
    comprehensive reward shaping for speedrunning behavior.
    """
    
    def __init__(self, config):
        super(PokemonRedEnv, self).__init__()
        
        # Store configuration
        self.config = config
        self.rom_path = config.get('rom_path', 'PokemonRed.gb')
        self.headless = config.get('headless', True)
        self.action_freq = config.get('action_freq', 24)  # Frames per action
        self.max_steps = config.get('max_steps', 8192)
        self.save_path = config.get('save_path', './session')
        self.save_screenshots = config.get('save_screenshots', True)
        
        # Verify ROM exists
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(f"Pokemon Red ROM not found at: {self.rom_path}")
        
        # Initialize PyBoy emulator
        window_type = 'null' if self.headless else 'SDL2'
        self.pyboy = PyBoy(
            self.rom_path,
            window=window_type
        )
        
        # Set emulation speed to maximum (no frame rate limit)
        self.pyboy.set_emulation_speed(0)
        
        # Get screen interface (updated API for PyBoy 2.0)
        self.screen = self.pyboy.screen
        
        # Define action space: D-pad (4) + A/B (2) + Start/Select (2) + No-op (1)
        self.action_space = spaces.Discrete(9)
        
        # Action mapping to PyBoy button names (PyBoy 2.0 uses strings)
        self.action_map = [
            'down',    # 0
            'left',    # 1
            'right',   # 2
            'up',      # 3
            'a',       # 4
            'b',       # 5
            'start',   # 6
            'select',  # 7
            None       # 8 - no operation
        ]
        
        # Define observation space: downsampled RGB screen
        # Original Game Boy screen: 144x160
        # Downsampled to: 120x128 for efficiency
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(120, 128, 3),
            dtype=np.uint8
        )
        
        # Initialize milestone tracking (must be before reset())
        self.milestones_achieved = set()
        
        # Initialize screen state tracker for advanced stagnation detection
        self.screen_tracker = ScreenStateTracker(
            history_size=100,  # Remember last 100 screens
            short_term_size=20  # Check for loops in last 20 screens
        )
        
        # Create save directories
        self.screenshots_dir = Path(self.save_path) / 'screenshots'
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables by calling reset
        self.reset()
        
        print(f"Environment initialized: {self.rom_path}")
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space.shape}")
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        Returns initial observation and info dict.
        
        Starts from ROM boot - agent must learn to navigate menus!
        """
        super().reset(seed=seed)
        
        # Soft reset PyBoy (restarts ROM from beginning)
        # This resets the game to the title screen
        for _ in range(60):  # Run a few frames to stabilize
            self.pyboy.tick()
        
        # Initialize step counter
        self.current_step = 0
        
        # Reset screen state tracker (but keep long-term memory for novelty detection)
        self.screen_tracker.reset()
        
        # Action tracking for info only
        self.action_history = deque(maxlen=10)  # Last 10 actions taken
        self.last_action = None
        
        # Initialize simple reward tracking
        self.episode_rewards = {
            'exploration': 0,     # Screen novelty and action bonuses
            'time_penalty': 0,    # Step cost
            'total': 0
        }
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one action in the environment.
        
        Args:
            action: Integer from 0-8 representing button press
            
        Returns:
            observation: Current screen state
            reward: Reward for this step
            terminated: Whether episode is finished
            truncated: Whether episode exceeded max steps
            info: Additional information dictionary
        """
        # Track action for diversity rewards
        self.action_history.append(action)
        self.last_action = action
        
        # Execute action in emulator
        self._take_action(action)
        
        # Get current observation
        observation = self._get_observation()
        
        # Calculate reward components
        reward = self._calculate_reward()
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get info dictionary
        info = self._get_info()
        
        # Check for milestones and take screenshots
        if self.save_screenshots:
            self._check_milestones()
        
        return observation, reward, terminated, truncated, info
    
    def _take_action(self, action):
        """
        Execute action in PyBoy emulator for action_freq frames.
        
        Actions are held for 8 frames then released for remaining frames.
        This mimics natural button presses and prevents stuck inputs.
        """
        button = self.action_map[action]
        
        # Press button (if not no-op)
        if button is not None:
            self.pyboy.button_press(button)
        
        # Execute frames with button held (8 frames)
        for _ in range(8):
            self.pyboy.tick()
        
        # Release button (if not no-op)
        if button is not None:
            self.pyboy.button_release(button)
        
        # Execute remaining frames (action_freq - 8 frames)
        for _ in range(self.action_freq - 8):
            self.pyboy.tick()
    
    def _get_observation(self):
        """
        Get current screen observation from emulator.
        
        Returns downsampled RGB image of Game Boy screen.
        """
        # Get raw screen array (144, 160, 3) - PyBoy 2.0 API
        # Convert PIL Image to RGB numpy array (removes alpha channel if present)
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen_array = np.asarray(screen_image)
        
        # Downsample to (120, 128, 3) for efficiency
        # Using skimage resize with anti-aliasing
        downsampled = resize(
            screen_array,
            (120, 128),
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.uint8)
        
        return downsampled
    
    def _get_screen_hash(self):
        """Get hash of current screen to detect if visuals changed."""
        # Get screen as bytes and hash it
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen_array = np.asarray(screen_image)
        return hash(screen_array.tobytes())
    
    def _read_memory(self, address):
        """Read single byte from Game Boy memory."""
        return self.pyboy.memory[address]
    
    def _read_uint16(self, address):
        """Read 16-bit unsigned integer (little-endian)."""
        low = self._read_memory(address)
        high = self._read_memory(address + 1)
        return low | (high << 8)
    
    def _read_bcd(self, address, num_bytes):
        """
        Read Binary Coded Decimal value.
        Used for money and some other counters in Pokemon Red.
        """
        value = 0
        for i in range(num_bytes):
            byte_val = self._read_memory(address + i)
            value = value * 100 + (byte_val >> 4) * 10 + (byte_val & 0x0F)
        return value
    
    def _count_bits(self, start_address, num_bytes):
        """Count number of set bits in a memory region."""
        count = 0
        for i in range(num_bytes):
            byte_val = self._read_memory(start_address + i)
            # Count bits using Brian Kernighan's algorithm
            while byte_val:
                byte_val &= byte_val - 1
                count += 1
        return count
    
    def _calculate_reward(self):
        """
        UNIVERSAL GAME BOY GAME LEARNER REWARD STRUCTURE
        
        Philosophy: Reward visual novelty and exploration, nothing game-specific.
        This approach works for ANY Game Boy game - Pokemon, Mario, Zelda, etc.
        
        Reward Components (SIMPLIFIED):
        1. Screen Novelty: Seeing new visual states (+1.0)
        2. Time Penalty: Small cost per step (-0.01) 
        3. Movement Bonus: Tiny reward for any action (prevents getting stuck)
        
        NO GAME-SPECIFIC REWARDS:
        - No Pokemon catching, battles, badges, etc.
        - No memory address reading for game state
        - No complex penalties or diversity requirements
        
        The agent learns to explore and discover game mechanics naturally!
        """
        total_reward = 0.0
        
        # ====================================================================
        # 1. TIME PENALTY - Encourage taking actions
        # ====================================================================
        time_penalty = -0.01  # Small cost per step to encourage progress
        total_reward += time_penalty
        self.episode_rewards['time_penalty'] = self.episode_rewards.get('time_penalty', 0) + time_penalty
        
        # ====================================================================
        # 2. SCREEN NOVELTY REWARD - Universal exploration incentive
        # ====================================================================
        # Get current screen for novelty detection
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen_array = np.asarray(screen_image)
        
        # Update screen tracker and get novelty reward
        screen_analysis = self.screen_tracker.update(screen_array)
        
        # Reward for seeing new screens (universal across all games)
        if screen_analysis.get('is_new_screen', False):
            novelty_reward = 1.0  # Simple, consistent reward for visual novelty
            total_reward += novelty_reward
            self.episode_rewards['exploration'] = self.episode_rewards.get('exploration', 0) + novelty_reward
        
        # Small bonus for screen diversity (prevents getting stuck in loops)
        diversity_bonus = screen_analysis.get('diversity_score', 0) * 0.1
        total_reward += diversity_bonus
        self.episode_rewards['exploration'] = self.episode_rewards.get('exploration', 0) + diversity_bonus
        
        # ====================================================================
        # 3. ACTION BONUS - Tiny reward for taking any action
        # ====================================================================
        # Prevents agent from learning to do nothing
        if self.last_action is not None and self.last_action != 8:  # 8 = no-op
            action_bonus = 0.01  # Tiny bonus for any non-no-op action
            total_reward += action_bonus
            self.episode_rewards['exploration'] = self.episode_rewards.get('exploration', 0) + action_bonus
        
        # ====================================================================
        # Track total episode reward
        # ====================================================================
        self.episode_rewards['total'] = self.episode_rewards.get('total', 0) + total_reward
        
        return total_reward
    
    def _check_termination(self):
        """
        Check if episode should terminate.
        
        Termination conditions:
        1. Natural game termination (if detectable)
        2. Episode length limit (handled by max_steps)
        
        For a universal game learner, we keep termination minimal.
        """
        # For now, let episodes run until max_steps
        # Game-specific termination can be added later if needed
        return False
    
    def _get_info(self):
        """
        Return information dictionary for monitoring.
        Simplified for universal game learning.
        """
        return {
            'step': self.current_step,
            'episode_reward': self.episode_rewards['total'],
            'screens_seen': len(self.screen_tracker.seen_screens) if hasattr(self.screen_tracker, 'seen_screens') else 0,
            'reward_breakdown': self.episode_rewards
        }
    
    def _check_milestones(self):
        """
        Check for milestone completion and capture screenshots.
        Simplified for universal game learning - just capture periodic screenshots.
        """
        # Take periodic screenshots for monitoring (every 1000 steps)
        if self.current_step % 1000 == 0 and self.current_step > 0:
            self._capture_milestone(f'step_{self.current_step}')
    
    def _capture_milestone(self, milestone_key):
        """
        Capture screenshot for monitoring.
        """
        # Get current screen (PyBoy 2.0 API) - convert to RGB
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen = np.asarray(screen_image)
        
        # Save screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{milestone_key}_{timestamp}.png"
        filepath = self.screenshots_dir / filename
        
        img = Image.fromarray(screen)
        img.save(filepath)
        
        print(f"\nScreenshot saved: {filepath}")
        print(f"   Step: {self.current_step}")
        print(f"   Total Reward: {self.episode_rewards['total']:.2f}")
    
    def close(self):
        """Clean up PyBoy emulator."""
        self.pyboy.stop()
    
    def render(self):
        """Render current state (for visualization)."""
        if not self.headless:
            # PyBoy handles rendering automatically in SDL2 mode
            pass
        return self._get_observation()

# ============================================================================
# ENVIRONMENT FACTORY - For parallel training
# ============================================================================

def make_env(rank, config):
    """
    Create environment factory function for parallel training.
    
    Args:
        rank: Environment ID for seeding
        config: Environment configuration dictionary
        
    Returns:
        Function that creates a new environment
    """
    def _init():
        # Add rank-specific save path
        env_config = config.copy()
        env_config['save_path'] = f"{config['save_path']}/env_{rank}"
        
        # Create environment
        env = PokemonRedEnv(env_config)
        
        # Wrap with Monitor for logging
        env = Monitor(env)
        
        return env
    
    return _init

# ============================================================================
# CUSTOM CALLBACKS - Milestone tracking and weight saving
# ============================================================================

class MilestoneCallback(BaseCallback):
    """
    Custom callback for tracking training progress and saving weights.
    
    This callback:
    1. Saves model checkpoints periodically
    2. Logs comprehensive training metrics to TensorBoard
    3. Tracks best model based on rewards
    4. Logs all reward components for analysis
    """
    
    def __init__(self, save_freq, save_path, verbose=1):
        super(MilestoneCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Real-time tracking for display
        self.last_print_step = 0
        self.recent_rewards = deque(maxlen=100)  # Last 100 episode rewards
        self.recent_coords = deque(maxlen=100)  # Last 100 coord counts
        
        # ACTION TRACKING - Critical for debugging!
        self.action_counts = [0] * 9  # Count of each action taken
        self.recent_actions = deque(maxlen=1000)  # Last 1000 actions
        self.action_names = ['DOWN', 'LEFT', 'RIGHT', 'UP', 'A', 'B', 'START', 'SELECT', 'WAIT']
        
        # Tracking for simplified reward analysis
        self.reward_components = {
            'exploration': [],
            'time_penalty': []
        }
        
        # Tracking for visualization
        self.training_history = {
            'timesteps': [],
            'mean_reward': [],
            'episode_length': [],
            'exploration': [],
            'badges': [],
            'pokemon_caught': []
        }
    
    def _on_step(self):
        """Called after each environment step."""
        # Track actions taken in this step
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if hasattr(actions, '__iter__'):
                for action in actions:
                    action_int = int(action)
                    self.action_counts[action_int] += 1
                    self.recent_actions.append(action_int)
            else:
                action_int = int(actions)
                self.action_counts[action_int] += 1
                self.recent_actions.append(action_int)
        
        # Save checkpoint periodically
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self.save_path / f"checkpoint_{self.n_calls}.zip"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Print real-time reward stats every 50 steps
        if self.n_calls - self.last_print_step >= 50:
            self.last_print_step = self.n_calls
            if len(self.recent_rewards) > 0:
                mean_r = np.mean(self.recent_rewards)
                max_r = np.max(self.recent_rewards)
                min_r = np.min(self.recent_rewards)
                mean_coords = np.mean(self.recent_coords) if len(self.recent_coords) > 0 else 0
                
                # Calculate action distribution from recent actions
                recent_action_dist = ""
                if len(self.recent_actions) > 0:
                    from collections import Counter
                    action_counter = Counter(self.recent_actions)
                    total = len(self.recent_actions)
                    # Show top 3 most common actions
                    top_actions = action_counter.most_common(3)
                    recent_action_dist = " | Actions: " + ", ".join([f"{self.action_names[a]}:{c*100/total:.0f}%" for a, c in top_actions])
                
                print(f"[Step {self.n_calls:6d}] Reward: {mean_r:+7.2f} (min:{min_r:+6.1f}, max:{max_r:+6.1f}) | Coords: {mean_coords:5.1f}{recent_action_dist}")
        
        # Log reward components from the last step if available
        if hasattr(self.locals.get('infos', [{}])[0], '__iter__'):
            for info in self.locals.get('infos', []):
                if 'episode' in info:
                    # Track recent episode stats
                    self.recent_rewards.append(info['episode']['r'])
                    if 'coordinates_explored' in info:
                        self.recent_coords.append(info['coordinates_explored'])
                    
                    # Log episode rewards breakdown if available
                    if 'reward_breakdown' in info:
                        breakdown = info['reward_breakdown']
                        for component, value in breakdown.items():
                            if component in self.reward_components:
                                self.reward_components[component].append(value)
        
        return True
    
    def _on_rollout_end(self):
        """Called at end of rollout (after collecting experiences)."""
        # Get episode statistics from environments
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
            
            # Update training history
            self.training_history['timesteps'].append(self.num_timesteps)
            self.training_history['mean_reward'].append(mean_reward)
            self.training_history['episode_length'].append(mean_length)
            
            # ===== LOG TO TENSORBOARD =====
            # Log basic metrics
            self.logger.record('custom/mean_episode_reward', mean_reward)
            self.logger.record('custom/mean_episode_length', mean_length)
            self.logger.record('custom/best_mean_reward', self.best_mean_reward)
            
            # Log reward components (averages over recent episodes)
            for component, values in self.reward_components.items():
                if len(values) > 0:
                    mean_value = np.mean(values[-100:])  # Last 100 values
                    self.logger.record(f'rewards/{component}', mean_value)
            
            # Log game-specific metrics from environment info
            if len(self.model.ep_info_buffer) > 0:
                # Try to extract custom metrics from episode info
                recent_episodes = list(self.model.ep_info_buffer)[-10:]  # Last 10 episodes
                
                # Initialize aggregators
                badges_total = []
                pokemon_caught_total = []
                pokemon_seen_total = []
                battles_won_total = []
                deaths_total = []
                coordinates_explored_total = []
                
                for ep_info in recent_episodes:
                    # Check if custom info exists
                    if 'badges' in ep_info:
                        badges_total.append(ep_info['badges'])
                    if 'pokemon_caught' in ep_info:
                        pokemon_caught_total.append(ep_info['pokemon_caught'])
                    if 'pokemon_seen' in ep_info:
                        pokemon_seen_total.append(ep_info['pokemon_seen'])
                    if 'battles_won' in ep_info:
                        battles_won_total.append(ep_info['battles_won'])
                    if 'deaths' in ep_info:
                        deaths_total.append(ep_info['deaths'])
                    if 'coordinates_explored' in ep_info:
                        coordinates_explored_total.append(ep_info['coordinates_explored'])
                
                # Log game progress metrics
                if len(badges_total) > 0:
                    self.logger.record('game/badges', np.mean(badges_total))
                if len(pokemon_caught_total) > 0:
                    self.logger.record('game/pokemon_caught', np.mean(pokemon_caught_total))
                if len(pokemon_seen_total) > 0:
                    self.logger.record('game/pokemon_seen', np.mean(pokemon_seen_total))
                if len(battles_won_total) > 0:
                    self.logger.record('game/battles_won', np.mean(battles_won_total))
                if len(deaths_total) > 0:
                    self.logger.record('game/deaths', np.mean(deaths_total))
                if len(coordinates_explored_total) > 0:
                    self.logger.record('game/coordinates_explored', np.mean(coordinates_explored_total))
            
            # Log action distribution
            if sum(self.action_counts) > 0:
                for i, count in enumerate(self.action_counts):
                    percentage = count / sum(self.action_counts) * 100
                    self.logger.record(f'actions/{self.action_names[i]}', percentage)
                
                # Log action diversity (entropy)
                total = sum(self.action_counts)
                probs = [c / total for c in self.action_counts if c > 0]
                action_entropy = -sum([p * np.log(p) for p in probs if p > 0])
                self.logger.record('actions/entropy', action_entropy)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = self.save_path / "best_model.zip"
                self.model.save(best_model_path)
                if self.verbose > 0:
                    print(f"\nNew best model! Mean reward: {mean_reward:.2f}")
                    print(f"   Saved to: {best_model_path}")
            
            # Log metrics to console
            if self.verbose > 0:
                print(f"\nTraining Update:")
                print(f"  Timesteps: {self.num_timesteps}")
                print(f"  Mean Reward: {mean_reward:.2f}")
                print(f"  Mean Episode Length: {mean_length:.0f}")


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Verify ROM file exists
ROM_PATH = 'PokemonRed.gb'
if not os.path.exists(ROM_PATH):
    print(f"\nERROR: Pokemon Red ROM not found at: {ROM_PATH}")
    print("Please place 'PokemonRed.gb' in the current directory.")
    print("The ROM should be exactly 1MB (1,048,576 bytes)")
    exit(1)

# Create session directory with timestamp
SESSION_NAME = f"pokemon_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SESSION_PATH = Path('./sessions') / SESSION_NAME
SESSION_PATH.mkdir(parents=True, exist_ok=True)

print(f"\nSession Directory: {SESSION_PATH}")

# Environment configuration
ENV_CONFIG = {
    'rom_path': ROM_PATH,
    'headless': True,  # No visualization during training (faster)
    'action_freq': 24,  # Execute 24 frames per action
    'max_steps': 8192,  # Maximum steps per episode
    'save_path': str(SESSION_PATH),
    'save_screenshots': True
}

def main():
    """Main training function."""
    # Training hyperparameters - BALANCED LEARNING SETTINGS
    NUM_ENVS = min(4, os.cpu_count())  # Parallel environments
    TOTAL_TIMESTEPS = 100_000  # Learn game progression
    SAVE_FREQ = 5_000  # Save checkpoint every 5k steps
    LEARNING_RATE = 0.001
    N_STEPS = 256  # ⬆️ INCREASED from 64 - More stable updates every 256*4=1024 steps
    BATCH_SIZE = 256  # Match N_STEPS for efficient learning
    N_EPOCHS = 4  # ⬇️ REDUCED from 8 - Prevent overfitting to random exploration
    GAMMA = 0.99  # ⬆️ INCREASED from 0.9 - Long-term strategy important for Pokemon
    GAE_LAMBDA = 0.95  # ⬆️ INCREASED from 0.85 - Better advantage estimation

    print(f"\nTraining Configuration:")
    print(f"  Parallel Environments: {NUM_ENVS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Steps per Update: {N_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gamma (discount): {GAMMA}")
    print(f"  GAE Lambda: {GAE_LAMBDA}")

    # ============================================================================
    # MAIN TRAINING LOOP
    # ============================================================================

    print(f"\nStarting Training...")
    print("=" * 80)

    # Create parallel environments
    print(f"Creating {NUM_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i, ENV_CONFIG) for i in range(NUM_ENVS)])

    print("Initializing PPO model...")
    # Create PPO model with CNN policy - BALANCED LEARNING HYPERPARAMETERS
    model = PPO(
        'CnnPolicy',
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=0.2,  # ⬇️ REDUCED from 0.4 - more conservative policy updates
        clip_range_vf=None,
        ent_coef=0.25,  # ⬆️ INCREASED from 0.15 - start higher to prevent collapse
        # This allows early exploration but doesn't overwhelm game rewards
        # 0.25 is moderate-high - enough exploration without entropy addiction
        # Will be reduced over time via callback to allow strategy development
        vf_coef=0.5,  # ⬆️ INCREASED from 0.3 - value function helps learn game strategy
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,  # Remove KL limit - allow reasonable policy changes
        tensorboard_log=str(SESSION_PATH / 'tensorboard'),
        policy_kwargs=dict(
            features_extractor_kwargs=dict(features_dim=512)
        ),
        verbose=1,
        seed=None,
        device='auto'
    )

    print(f"Model initialized. Device: {model.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Create entropy scheduling callback to prevent policy collapse
    class EntropyScheduler(BaseCallback):
        """Gradually reduce entropy coefficient to allow strategy development while preventing collapse."""
        
        def __init__(self, initial_ent_coef=0.25, final_ent_coef=0.05, total_timesteps=100000):
            super().__init__()
            self.initial_ent_coef = initial_ent_coef
            self.final_ent_coef = final_ent_coef
            self.total_timesteps = total_timesteps
            
        def _on_step(self):
            # Calculate current progress (0 to 1)
            progress = min(self.num_timesteps / self.total_timesteps, 1.0)
            
            # Linear decay from initial to final entropy coefficient
            current_ent_coef = self.initial_ent_coef - (self.initial_ent_coef - self.final_ent_coef) * progress
            
            # Update the model's entropy coefficient
            self.model.ent_coef = current_ent_coef
            
            # Log the current entropy coefficient every 1000 steps
            if self.num_timesteps % 1000 == 0:
                self.logger.record('entropy_scheduler/ent_coef', current_ent_coef)
                
            return True

    # Create callbacks
    milestone_callback = MilestoneCallback(
        save_freq=SAVE_FREQ,
        save_path=SESSION_PATH / 'checkpoints',
        verbose=1
    )
    
    entropy_callback = EntropyScheduler(
        initial_ent_coef=0.25,
        final_ent_coef=0.05, 
        total_timesteps=TOTAL_TIMESTEPS
    )

    # Start training
    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[milestone_callback, entropy_callback],  # Use both callbacks
            log_interval=1,  # Log every update for more frequent TensorBoard data
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Training complete
    end_time = time.time()
    training_duration = end_time - start_time

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Duration: {training_duration / 3600:.2f} hours")
    print(f"Final model saved to: {SESSION_PATH}")

    # Save final model
    final_model_path = SESSION_PATH / 'final_model.zip'
    model.save(final_model_path)
    print(f"Final model: {final_model_path}")

    # Save training history
    history_path = SESSION_PATH / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(callback.training_history, f, indent=2)
    print(f"Training history: {history_path}")

    # Close environments
    env.close()

    print("\nTraining session complete!")
    print(f"View results with: tensorboard --logdir {SESSION_PATH / 'tensorboard'}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    # This is required for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    
    main()