"""
Pokemon Red Reinforcement Learning Speedrunning Agent
Main Training Implementation with Comprehensive Reward Shaping

This implementation creates a PPO-based agent that learns to speedrun Pokemon Red
through the Elite Four, with detailed reward engineering and swarm-based weight sharing.

Requirements:
- Pokemon Red ROM file named 'PokemonRed.gb' in same directory
- PyBoy emulator for Game Boy emulation
- Stable-baselines3 for PPO algorithm
- PyTorch as backend
- Gymnasium for environment interface

Architecture:
- Custom Gymnasium environment wrapping PyBoy emulator
- Multi-component reward system with progress tracking
- Coordinate-based exploration (memory efficient)
- Parallel environment training for speed
- Automatic milestone screenshot capture
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

print("=" * 80)
print("POKEMON RED RL SPEEDRUNNING AGENT - TRAINING SYSTEM")
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
        
        # Initialize position tracking for exploration
        self.visited_coordinates = set()
        self.visited_maps = set()  # Track unique maps visited
        self.last_position = None
        self.position_stuck_counter = 0
        
        # Screen hash tracking for TRUE stagnation detection
        self.screen_hash_history = deque(maxlen=20)  # Last 20 screen hashes
        self.last_screen_hash = None
        self.same_screen_counter = 0
        
        # Action diversity tracking
        self.action_history = deque(maxlen=10)  # Last 10 actions taken
        self.last_action = None
        
        # Initialize menu tracking
        self.menu_time_counter = 0
        self.last_menu_state = 0
        
        # Initialize battle tracking
        self.last_battle_type = 0
        self.battles_won = 0
        self.battles_lost = 0
        self.damage_dealt_total = 0
        self.last_enemy_hp = 0
        
        # Initialize Pokemon tracking
        self.pokemon_seen_count = 0
        self.pokemon_caught_count = 0
        self.last_seen_count = 0
        self.last_caught_count = 0
        
        # Initialize badge tracking
        self.badges_obtained = 0
        self.last_badge_count = 0
        
        # Initialize level tracking
        self.party_level_sum = 0
        self.max_opponent_level = 0
        
        # Initialize death tracking
        self.deaths = 0
        
        # Initialize event tracking
        self.events_completed = 0
        
        # Initialize reward tracking for visualization
        self.episode_rewards = {
            'exploration': 0,
            'battle_won': 0,
            'battle_engaged': 0,
            'damage_dealt': 0,
            'pokemon_caught': 0,
            'gym_badge': 0,
            'death_penalty': 0,
            'stuck_penalty': 0,
            'menu_penalty': 0,
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
        Calculate comprehensive reward based on multiple components.
        
        EXPLORATION-FOCUSED REWARD STRUCTURE (based on P2 findings):
        - Much higher rewards for discovery and progress
        - Lower penalties to encourage experimentation
        
        Reward Components:
        1. Exploration: New coordinates visited (+20.0 per new location) ⬆️
        2. Map Discovery: New map areas (+100.0) ⬆️
        3. Battle Engagement: Starting battles (+5.0) ⬆️
        4. Damage Dealt: Damaging enemy Pokemon (+2.0 per HP) ⬆️
        5. Battle Victory: Winning battles (+200.0) ⬆️
        6. Pokemon Catching: Catching new Pokemon (+400.0) ⬆️
        7. Gym Badges: Obtaining badges (+500.0) ⬆️
        8. Death Penalty: Pokemon fainting (-5.0)
        9. Stuck Penalty: Staying in same position (-0.2 per step) ⬇️
        10. Menu Penalty: Staying in menus too long (-0.01 per step) ⬇️
        11. Time Penalty: Per-step cost (-0.01) ⬇️
        """
        total_reward = 0.0
        
        # Apply small time penalty to encourage action
        time_penalty = -0.01
        total_reward += time_penalty
        
        # ====================================================================
        # 0. SCREEN STAGNATION DETECTION - Critical for title screen!
        # ====================================================================
        current_screen_hash = self._get_screen_hash()
        
        # Check if screen has changed at all
        if current_screen_hash == self.last_screen_hash:
            self.same_screen_counter += 1
            # HARSH penalty for being on same screen
            if self.same_screen_counter > 5:  # Just 5 steps of same screen
                stagnation_penalty = -1.0  # HUGE penalty!
                total_reward += stagnation_penalty
                self.episode_rewards['stuck_penalty'] += stagnation_penalty
        else:
            # Screen changed! Small reward
            screen_change_reward = 0.2
            total_reward += screen_change_reward
            self.episode_rewards['exploration'] += screen_change_reward
            self.same_screen_counter = 0
        
        self.screen_hash_history.append(current_screen_hash)
        self.last_screen_hash = current_screen_hash
        
        # ====================================================================
        # 0.5. ACTION DIVERSITY REWARD - Encourage trying different buttons
        # ====================================================================
        if len(self.action_history) >= 3:
            # Count unique actions in recent history
            unique_actions = len(set(self.action_history))
            # Reward diversity
            if unique_actions >= 3:  # Using at least 3 different buttons
                diversity_bonus = 0.1
                total_reward += diversity_bonus
                self.episode_rewards['exploration'] += diversity_bonus
            elif unique_actions == 1:  # Spamming same button
                diversity_penalty = -0.2
                total_reward += diversity_penalty
                self.episode_rewards['stuck_penalty'] += diversity_penalty
        
        # ====================================================================
        # 1. EXPLORATION REWARD - New coordinates visited
        # ====================================================================
        current_x = self._read_memory(PLAYER_X_ADDRESS)
        current_y = self._read_memory(PLAYER_Y_ADDRESS)
        current_map = self._read_memory(MAP_ID_ADDRESS)
        current_pos = (current_map, current_x, current_y)
        
        # SEVERE penalty if stuck on title screen (map 0, pos 0,0)
        if current_map == 0 and current_x == 0 and current_y == 0:
            title_screen_penalty = -0.1  # Continuous penalty every step on title screen
            total_reward += title_screen_penalty
            self.episode_rewards['menu_penalty'] += title_screen_penalty
        
        # Check if this is a new position
        if current_pos not in self.visited_coordinates:
            self.visited_coordinates.add(current_pos)
            exploration_reward = 0.5  # ⬇️ Reduced - save bigger rewards for actual progress
            total_reward += exploration_reward
            self.episode_rewards['exploration'] += exploration_reward
        
        # IMPORTANT: Reward ANY movement to encourage constant action
        if self.last_position is not None and self.last_position != current_pos:
            movement_reward = 0.5  # ⬆️ Reward for moving at all!
            total_reward += movement_reward
            self.episode_rewards['exploration'] += movement_reward
        
        # Track new maps discovered
        if current_map not in self.visited_maps:
            self.visited_maps.add(current_map)
            map_discovery_reward = 100.0  # ⬆️ Huge reward for discovering new areas
            total_reward += map_discovery_reward
            self.episode_rewards['exploration'] += map_discovery_reward
            print(f"🗺️  NEW MAP DISCOVERED! ID: {current_map}")
        
        # Check if player is stuck in same position
        if self.last_position == current_pos:
            self.position_stuck_counter += 1
            # Apply penalty immediately for being stuck
            if self.position_stuck_counter > 3:  # ⬇️ Reduced threshold from 10 to 3
                stuck_penalty = -0.5  # ⬆️ Increased penalty from -0.2 to discourage standing still
                total_reward += stuck_penalty
                self.episode_rewards['stuck_penalty'] += stuck_penalty
        else:
            self.position_stuck_counter = 0
        
        self.last_position = current_pos
        
        # ====================================================================
        # 2. MENU PENALTY - Discourage staying in menus
        # ====================================================================
        current_menu = self._read_memory(MENU_STATE_ADDRESS)
        
        # If in a menu (non-zero menu state)
        if current_menu != 0:
            self.menu_time_counter += 1
            # CONTINUOUS penalty for being in menu - gets worse over time
            menu_penalty = -0.05 * (1 + self.menu_time_counter / 100.0)  # Escalating penalty
            total_reward += menu_penalty
            self.episode_rewards['menu_penalty'] += menu_penalty
        else:
            self.menu_time_counter = 0
        
        # ====================================================================
        # 3. BATTLE ENGAGEMENT REWARD
        # ====================================================================
        current_battle = self._read_memory(IN_BATTLE_ADDRESS)
        
        # Check if entering a battle (transition from 0 to non-zero)
        if current_battle != 0 and self.last_battle_type == 0:
            battle_engage_reward = 5.0  # ⬆️ Increased from 0.5 to encourage battles
            total_reward += battle_engage_reward
            self.episode_rewards['battle_engaged'] += battle_engage_reward
        
        self.last_battle_type = current_battle
        
        # ====================================================================
        # 4. DAMAGE DEALT REWARD
        # ====================================================================
        if current_battle != 0:
            # Read enemy HP (this is approximate - actual tracking is complex)
            enemy_hp = self._read_uint16(ENEMY_MON_LEVEL_ADDRESS + 10)
            
            # If enemy HP decreased, reward damage dealt
            if self.last_enemy_hp > 0 and enemy_hp < self.last_enemy_hp:
                damage = self.last_enemy_hp - enemy_hp
                damage_reward = damage * 2.0  # ⬆️ Increased from 0.1 for stronger battle incentive
                total_reward += damage_reward
                self.episode_rewards['damage_dealt'] += damage_reward
                self.damage_dealt_total += damage
            
            self.last_enemy_hp = enemy_hp
        else:
            self.last_enemy_hp = 0
        
        # ====================================================================
        # 5. BATTLE VICTORY REWARD
        # ====================================================================
        # Battle ended (transition from non-zero to zero)
        if current_battle == 0 and self.last_battle_type != 0:
            # Check if player party still has HP (won) vs all fainted (lost)
            party_alive = False
            for i in range(6):
                hp = self._read_uint16(PARTY_HP_START + i * 2)
                if hp > 0:
                    party_alive = True
                    break
            
            if party_alive:
                # Won the battle
                battle_win_reward = 200.0  # ⬆️ Increased from 2.0 - HUGE victory reward!
                total_reward += battle_win_reward
                self.episode_rewards['battle_won'] += battle_win_reward
                self.battles_won += 1
            else:
                # Lost the battle (death penalty applied below)
                self.battles_lost += 1
        
        # ====================================================================
        # 6. POKEMON CATCHING REWARD
        # ====================================================================
        # Count Pokemon owned via Pokedex bitfield
        current_caught = self._count_bits(POKEDEX_OWNED_START, 19)
        
        if current_caught > self.last_caught_count:
            new_catches = current_caught - self.last_caught_count
            catch_reward = new_catches * 400.0  # ⬆️ Increased from 5.0 - MASSIVE catch reward!
            total_reward += catch_reward
            self.episode_rewards['pokemon_caught'] += catch_reward
            self.pokemon_caught_count = current_caught
            self.last_caught_count = current_caught
        
        # Track Pokemon seen (for exploration metric)
        current_seen = self._count_bits(POKEDEX_SEEN_START, 19)
        if current_seen > self.last_seen_count:
            self.pokemon_seen_count = current_seen
            self.last_seen_count = current_seen
        
        # ====================================================================
        # 7. GYM BADGE REWARD
        # ====================================================================
        badge_flags = self._read_memory(BADGE_FLAGS_ADDRESS)
        current_badges = bin(badge_flags).count('1')
        
        if current_badges > self.last_badge_count:
            new_badges = current_badges - self.last_badge_count
            badge_reward = new_badges * 500.0  # ⬆️ Increased from 20.0 - HUGE badge reward!
            total_reward += badge_reward
            self.episode_rewards['gym_badge'] += badge_reward
            self.badges_obtained = current_badges
            self.last_badge_count = current_badges
        
        # ====================================================================
        # 8. DEATH PENALTY - All party Pokemon fainted
        # ====================================================================
        all_fainted = True
        for i in range(6):
            hp = self._read_uint16(PARTY_HP_START + i * 2)
            if hp > 0:
                all_fainted = False
                break
        
        if all_fainted and self.party_level_sum > 0:  # Ensure party exists
            death_penalty = -5.0
            total_reward += death_penalty
            self.episode_rewards['death_penalty'] += death_penalty
            self.deaths += 1
        
        # ====================================================================
        # Update party level sum for info tracking
        # ====================================================================
        party_size = self._read_memory(PARTY_SIZE_ADDRESS)
        level_sum = 0
        for i in range(min(party_size, 6)):
            level = self._read_memory(PARTY_LEVEL_START + i)
            level_sum += level
        self.party_level_sum = level_sum
        
        # ====================================================================
        # Track total episode reward
        # ====================================================================
        self.episode_rewards['total'] += total_reward
        
        return total_reward
    
    def _check_termination(self):
        """
        Check if episode should terminate.
        
        Termination conditions:
        1. Defeated Elite Four and Champion (WIN CONDITION)
        2. Blacked out too many times (> 10 deaths)
        """
        # Check for Elite Four completion (map ID or event flag)
        # This is simplified - actual implementation needs specific event flags
        
        # Check death limit
        if self.deaths > 10:
            print("Episode terminated: Too many deaths")
            return True
        
        return False
    
    def _get_info(self):
        """
        Return information dictionary for monitoring.
        """
        return {
            'step': self.current_step,
            'position': self.last_position,
            'coordinates_explored': len(self.visited_coordinates),
            'badges': self.badges_obtained,
            'party_level_sum': self.party_level_sum,
            'pokemon_caught': self.pokemon_caught_count,
            'pokemon_seen': self.pokemon_seen_count,
            'battles_won': self.battles_won,
            'battles_lost': self.battles_lost,
            'deaths': self.deaths,
            'episode_reward': self.episode_rewards['total'],
            'milestones': len(self.milestones_achieved),
            'reward_breakdown': self.episode_rewards  # 🔥 ADD REWARD BREAKDOWN FOR TENSORBOARD!
        }
    
    def _check_milestones(self):
        """
        Check for milestone completion and capture screenshots.
        """
        # Check badge milestones
        if self.badges_obtained >= 1 and 'first_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('first_gym_badge')
        if self.badges_obtained >= 2 and 'second_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('second_gym_badge')
        if self.badges_obtained >= 3 and 'third_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('third_gym_badge')
        if self.badges_obtained >= 4 and 'fourth_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('fourth_gym_badge')
        if self.badges_obtained >= 5 and 'fifth_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('fifth_gym_badge')
        if self.badges_obtained >= 6 and 'sixth_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('sixth_gym_badge')
        if self.badges_obtained >= 7 and 'seventh_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('seventh_gym_badge')
        if self.badges_obtained >= 8 and 'eighth_gym_badge' not in self.milestones_achieved:
            self._capture_milestone('eighth_gym_badge')
        
        # Check Pokemon catching milestones
        if self.pokemon_caught_count >= 1 and 'first_pokemon_caught' not in self.milestones_achieved:
            self._capture_milestone('first_pokemon_caught')
    
    def _capture_milestone(self, milestone_key):
        """
        Capture screenshot when milestone is achieved.
        """
        if milestone_key in self.milestones_achieved:
            return
        
        self.milestones_achieved.add(milestone_key)
        
        # Get current screen (PyBoy 2.0 API) - convert to RGB
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen = np.asarray(screen_image)
        
        # Save screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{milestone_key}_{timestamp}.png"
        filepath = self.screenshots_dir / filename
        
        img = Image.fromarray(screen)
        img.save(filepath)
        
        milestone_desc = MILESTONES.get(milestone_key, milestone_key)
        print(f"\nMILESTONE ACHIEVED: {milestone_desc}")
        print(f"   Screenshot saved: {filepath}")
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
        
        # Tracking for detailed reward analysis
        self.reward_components = {
            'exploration': [],
            'battle_won': [],
            'battle_engaged': [],
            'damage_dealt': [],
            'pokemon_caught': [],
            'gym_badge': [],
            'death_penalty': [],
            'stuck_penalty': [],
            'menu_penalty': []
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
        # Save checkpoint periodically
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self.save_path / f"checkpoint_{self.n_calls}.zip"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Log reward components from the last step if available
        if hasattr(self.locals.get('infos', [{}])[0], '__iter__'):
            for info in self.locals.get('infos', []):
                if 'episode' in info:
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
    # Training hyperparameters
    NUM_ENVS = min(3, os.cpu_count())  # Use up to 24 parallel environments
    TOTAL_TIMESTEPS = 50_000  # 100k steps
    SAVE_FREQ = 5_000  # Save checkpoint every 5k steps
    LEARNING_RATE = 3e-4
    N_STEPS = 2048  # Steps per environment before update
    BATCH_SIZE = 512
    N_EPOCHS = 1
    GAMMA = 0.999  # Discount factor
    GAE_LAMBDA = 0.95

    print(f"\nTraining Configuration:")
    print(f"  Parallel Environments: {NUM_ENVS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Steps per Update: {N_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")

    # ============================================================================
    # MAIN TRAINING LOOP
    # ============================================================================

    print(f"\nStarting Training...")
    print("=" * 80)

    # Create parallel environments
    print(f"Creating {NUM_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i, ENV_CONFIG) for i in range(NUM_ENVS)])

    print("Initializing PPO model...")
    # Create PPO model with CNN policy - EXPLORATION-FOCUSED HYPERPARAMETERS
    model = PPO(
        'CnnPolicy',
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=0.3,  # ⬆️ Increased from 0.2 - allows bigger policy changes for exploration
        clip_range_vf=None,
        ent_coef=0.05,  # ⬆️ Increased from 0.01 - MUCH stronger exploration bonus!
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
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

    # Create callback
    callback = MilestoneCallback(
        save_freq=SAVE_FREQ,
        save_path=SESSION_PATH / 'checkpoints',
        verbose=1
    )

    # Start training
    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
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