"""
Pokemon Red RL - Screen-Based Exploration Agent
Based on the successful approach from Peter Whidden's Pokemon Red RL project

This implementation follows the proven methodology:
- Primary reward: Screen-based novelty detection with pixel threshold
- Secondary reward: Combined Pokemon levels (only increases) 
- Grid-aligned movement (24 frame intervals)
- Screen comparison against historical record
- Simple, focused reward structure

CORE PHILOSOPHY (from the working system):
- Compare current screen against record of all seen screens
- Reward for unique screens (several hundred pixel difference threshold)
- Combined Pokemon levels as secondary progression signal
- Avoid complex game-specific rewards that cause issues
- Let exploration drive learning, not hand-crafted objectives

Requirements:
- Pokemon Red ROM (PokemonRed.gb)
- PyBoy emulator for Game Boy emulation
- Stable-baselines3 for PPO algorithm
- PyTorch as backend
- Gymnasium for environment interface

Architecture:
- Custom Gymnasium environment wrapping PyBoy emulator
- Screen record system for novelty detection
- Pixel-difference based exploration rewards
- Level-based progression rewards (increases only)
- Grid-aligned 24-frame action intervals
"""

import os
import warnings
import sys
import io
from contextlib import redirect_stderr

# Set TensorFlow environment variables before any TF imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow info and warning messages

# Suppress warnings that can be caught by Python's warning system
warnings.filterwarnings('ignore', message='Unable to preload all dependencies for SDL2_ttf')
warnings.filterwarnings('ignore', message='Unable to preload all dependencies for SDL2_image')
warnings.filterwarnings('ignore', message='Gym has been unmaintained since 2022')
warnings.filterwarnings('ignore', category=UserWarning, module='gym')

# Create a stderr filter to catch gym warnings that bypass the warning system
class StderrFilter:
    def __init__(self):
        self.original_stderr = sys.stderr
        self.buffer = io.StringIO()
        
    def write(self, text):
        # Filter out gym deprecation messages
        if "Gym has been unmaintained since 2022" not in text and \
           "Please upgrade to Gymnasium" not in text and \
           "Users of this version of Gym should be able to simply replace" not in text and \
           "See the migration guide" not in text:
            self.original_stderr.write(text)
            
    def flush(self):
        self.original_stderr.flush()
        
    def close(self):
        # Delegate close to original stderr
        if hasattr(self.original_stderr, 'close'):
            self.original_stderr.close()
            
    def __getattr__(self, name):
        # Delegate any other attributes to original stderr
        return getattr(self.original_stderr, name)

# Apply stderr filter during imports
sys.stderr = StderrFilter()

try:
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

    # Import stable-baselines3 for PPO algorithm (this triggers the gym warning)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    # Import preprocessing utilities
    from skimage.transform import resize

    # Import custom screen state tracker
    from screen_state_tracker import ScreenStateTracker

finally:
    # Restore original stderr
    sys.stderr = sys.stderr.original_stderr

print("=" * 80)
print("POKEMON RED RL - SCREEN-BASED EXPLORATION AGENT")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# ============================================================================
# SCREEN-BASED EXPLORATION SYSTEM
# Based on successful Pokemon Red RL methodology
# ============================================================================

class ScreenRecord:
    """
    Maintains a record of all unique screens seen by the AI.
    Implements the core novelty detection system from the successful approach.
    """
    
    def __init__(self, pixel_threshold=300):
        """
        Initialize screen record system.
        
        Args:
            pixel_threshold: Number of pixels that must differ to count as new screen
                           (original used "several hundred pixels")
        """
        self.seen_screens = []  # Store all unique screens as numpy arrays
        self.pixel_threshold = pixel_threshold
        self.total_unique_screens = 0
        
    def is_novel_screen(self, screen_array):
        """
        Check if current screen is novel compared to all previously seen screens.
        
        Args:
            screen_array: Current screen as numpy array (H, W, 3)
            
        Returns:
            bool: True if screen is novel (should be rewarded)
        """
        # Convert to grayscale for more stable comparison
        if len(screen_array.shape) == 3:
            screen_gray = np.mean(screen_array, axis=2)
        else:
            screen_gray = screen_array
            
        # Compare against all previously seen screens
        for seen_screen in self.seen_screens:
            if len(seen_screen.shape) == 3:
                seen_gray = np.mean(seen_screen, axis=2)
            else:
                seen_gray = seen_screen
                
            # Count different pixels
            diff_pixels = np.sum(np.abs(screen_gray - seen_gray) > 10)  # Small tolerance for noise
            
            # If difference is below threshold, screen is not novel
            if diff_pixels < self.pixel_threshold:
                return False
        
        # If we get here, screen is novel - add to record
        self.seen_screens.append(screen_gray.copy())
        self.total_unique_screens += 1
        return True
    
    def get_exploration_progress(self):
        """Get current exploration statistics."""
        return {
            'unique_screens': self.total_unique_screens,
            'total_comparisons': len(self.seen_screens)
        }

# ============================================================================
# SIMPLIFIED MEMORY ADDRESSES - Focus on Essential Game State
# ============================================================================

# Player position and map
PLAYER_X_ADDRESS = 0xD362
PLAYER_Y_ADDRESS = 0xD361
MAP_ID_ADDRESS = 0xD35E
PLAYER_NAME_ADDRESS = 0xD158          # Player name (7 bytes)

# Pokemon party - Essential for level-based rewards
PARTY_COUNT_ADDRESS = 0xD163          # Number of Pokemon in party (1 byte)

# Pokemon level addresses (simplified - just get the levels directly)
PARTY_POKEMON_1_LEVEL = 0xD16E        # 1st Pokemon level (offset +3 from start)
PARTY_POKEMON_2_LEVEL = 0xD19A        # 2nd Pokemon level  
PARTY_POKEMON_3_LEVEL = 0xD1C6        # 3rd Pokemon level
PARTY_POKEMON_4_LEVEL = 0xD1F2        # 4th Pokemon level
PARTY_POKEMON_5_LEVEL = 0xD21E        # 5th Pokemon level
PARTY_POKEMON_6_LEVEL = 0xD24A        # 6th Pokemon level

# Badge flags (for additional progress tracking)
BADGE_FLAGS_ADDRESS = 0xD356

# Battle state tracking
BATTLE_TYPE_ADDRESS = 0xD057          # 0 = not in battle, >0 = in battle

# ============================================================================
# SIMPLIFIED MILESTONES - Focus on major progression points
# ============================================================================

MILESTONES = {
    'first_pokemon': 'Got first Pokemon',
    'first_level_up': 'First Pokemon leveled up', 
    'first_evolution': 'First Pokemon evolution',
    'first_gym_badge': 'Defeated first gym leader',
    'viridian_forest': 'Entered Viridian Forest',
    'pewter_city': 'Reached Pewter City'
}

# ============================================================================
# CUSTOM GYMNASIUM ENVIRONMENT - Pokemon Red
# ============================================================================

class PokemonRedEnv(gym.Env):
    """
    Pokemon Red Environment following the proven screen-based exploration methodology.
    
    Key design principles from successful implementation:
    - Primary reward: Screen-based novelty detection  
    - Secondary reward: Combined Pokemon levels (increases only)
    - Grid-aligned movement (24 frame action intervals)
    - Simplified reward structure focused on exploration + progression
    - No complex game-specific rewards that cause issues
    
    Observation Space: (144, 160, 3) RGB image (full Game Boy screen)
    Action Space: Discrete(9) - 4 directions, 2 buttons, start, select, no-op
    """
    
    def __init__(self, config):
        super(PokemonRedEnv, self).__init__()
        
        # Store configuration
        self.config = config
        self.rom_path = config.get('rom_path', 'PokemonRed.gb')
        self.headless = config.get('headless', True)
        self.action_freq = 24  # FIXED: 24 frames per action for grid alignment
        self.max_steps = config.get('max_steps', 8192)
        self.save_path = config.get('save_path', './session')
        self.save_screenshots = config.get('save_screenshots', True)
        
        # Verify ROM exists
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(f"Pokemon Red ROM not found at: {self.rom_path}")
        
        # Initialize PyBoy emulator
        # Check if this is a demo mode that needs visuals
        force_visual = config.get('force_visual', False)
        
        # Force headless mode on Windows for training, but allow visuals for demos
        import platform
        if platform.system() == 'Windows' and not force_visual:
            self.headless = True
        
        window_type = 'null' if self.headless else 'SDL2'
        
        # Suppress SDL2 warnings
        import warnings
        warnings.filterwarnings('ignore', message='Unable to preload all dependencies for SDL2')
        
        # Configure SDL2 for better window visibility on Windows
        if not self.headless:
            os.environ['SDL_VIDEODRIVER'] = 'windows'  # Use Windows native driver
            os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'  # Position window
        
        self.pyboy = PyBoy(
            self.rom_path,
            window=window_type,
            debug=False  # Disable debug mode to reduce warnings
        )
        
        # Set emulation speed to maximum (no frame rate limit)
        self.pyboy.set_emulation_speed(0)
        
        # Get screen interface
        self.screen = self.pyboy.screen
        
        # Define action space: D-pad (4) + A/B (2) + Start/Select (2) + No-op (1)
        self.action_space = spaces.Discrete(9)
        
        # Action mapping to PyBoy button names
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
        
        # Define observation space following proven architecture:
        # - 3 most recent screens stacked (for short-term memory)
        # - Resolution reduced 4x (36x40 from 144x160) for speed/memory
        # - Status bars overlaid for game state info
        # - Grayscale for efficiency (1 channel per screen)
        self.reduced_height = 36  # 144 / 4
        self.reduced_width = 40   # 160 / 4
        self.num_stacked_screens = 3
        
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.reduced_height, self.reduced_width, self.num_stacked_screens),
            dtype=np.uint8
        )
        
        # Initialize screen history for stacking
        self.screen_history = deque(maxlen=self.num_stacked_screens)
        
        # Initialize screen-based exploration system
        self.screen_record = ScreenRecord(pixel_threshold=300)  # "several hundred pixels"
        
        # Initialize level tracking for progression rewards
        self.previous_total_levels = 0
        
        # Initialize milestone tracking
        self.milestones_achieved = set()
        
        # Create save directories - organize by env rank if provided
        env_rank = config.get('env_rank', 0)
        if env_rank > 0:
            self.screenshots_dir = Path(self.save_path) / f'screenshots_env_{env_rank}'
            self.cnn_debug_dir = Path(self.save_path) / f'cnn_debug_env_{env_rank}'
        else:
            self.screenshots_dir = Path(self.save_path) / 'screenshots'
            self.cnn_debug_dir = Path(self.save_path) / 'cnn_debug'
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.cnn_debug_dir.mkdir(parents=True, exist_ok=True)
        
        # CNN Input Debugging System - visualize what the model sees
        self.debug_cnn_input = config.get('debug_cnn_input', False)
        self.cnn_save_frequency = config.get('cnn_save_frequency', 24)  # Save every N steps
        self.step_count = 0
        
        # Initialize tracking variables by calling reset
        self.reset()
        
        print(f"Environment initialized: {self.rom_path}")
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space.shape}")
        print(f"Action frequency: {self.action_freq} frames (grid-aligned)")
        print(f"Screen novelty threshold: {self.screen_record.pixel_threshold} pixels")
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        Returns initial observation and info dict.
        
        Following proven methodology: start from ROM boot, let exploration rewards
        guide the agent through character creation naturally.
        """
        super().reset(seed=seed)
        
        # Soft reset PyBoy (restarts ROM from beginning)
        for _ in range(60):  # Run a few frames to stabilize
            self.pyboy.tick()
        
        # Initialize step counter
        self.current_step = 0
        
        # Initialize level tracking (but don't reset screen record between episodes!)
        self.previous_total_levels = 0
        
        # Initialize screen history for stacking (clear between episodes)
        self.screen_history.clear()
        
        # Initialize 7-reward system tracking variables (reset each episode)
        self.died_count = 0
        self.max_event_score = 0
        self.previous_badge_count = 0
        self.total_healing_rew = 0
        self.last_party_hp = 0
        self.max_op_level = 0
        
        # Action tracking for info
        self.action_history = deque(maxlen=10)
        self.last_action = None
        
        # Complete reward tracking - all 7 components plus total
        self.episode_rewards = {
            'event': 0,           # Event progress
            'level': 0,           # Pokemon levels  
            'heal': 0,            # Healing progress
            'op_lvl': 0,          # Opponent levels
            'dead': 0,            # Death penalty
            'badge': 0,           # Badge progress
            'explore': 0,         # Exploration (screen novelty)
            'total': 0
        }
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one action in the environment.
        
        Following proven methodology:
        - 24-frame action intervals for grid alignment
        - Screen-based exploration as primary reward
        - Level-based progression as secondary reward
        """
        # Track action
        self.action_history.append(action)
        self.last_action = action
        
        # Execute action in emulator (24 frames for grid alignment)
        self._take_action(action)
        
        # Get full resolution screen for reward calculation (novelty detection)
        screen_image = self.pyboy.screen.image.convert('RGB')
        full_screen_array = np.asarray(screen_image)
        
        # Calculate reward using full resolution screen
        reward = self._calculate_reward(full_screen_array)
        
        # Track deaths (Pokemon fainting)
        self._track_deaths()
        
        # Get reduced/stacked observation for policy network
        observation = self._get_observation()
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Get info dictionary
        info = self._get_info()
        
        # Save CNN debug visualization (what the model actually sees)
        self.step_count += 1
        self._save_cnn_debug_visualization(observation, self.step_count, action, reward)
        
        # Take periodic screenshots for monitoring
        if self.save_screenshots and self.current_step % 1000 == 0:
            self._capture_screenshot()
        
        return observation, reward, terminated, truncated, info
    
    def _take_action(self, action):
        """
        Execute action in PyBoy emulator for exactly 24 frames.
        
        Critical: 24 frames = exactly enough time to move one grid space.
        This ensures grid-aligned movement and makes screen comparison more effective.
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
        
        # Execute remaining frames (24 - 8 = 16 frames)
        for _ in range(16):
            self.pyboy.tick()
    
    def _get_observation(self):
        """
        Get observation following proven architecture:
        - 3 most recent screens stacked for short-term memory
        - Resolution reduced 4x for speed/memory efficiency  
        - Grayscale conversion
        - Visual status bars overlaid for game state
        """
        # Get raw screen from PyBoy
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen_array = np.asarray(screen_image)
        
        # Convert to grayscale and reduce resolution 4x
        screen_gray = np.mean(screen_array, axis=2).astype(np.uint8)  # RGB to grayscale
        screen_reduced = screen_gray[::4, ::4]  # Reduce from 144x160 to 36x40
        
        # Add status bars (simple visual indicators for game state)
        screen_with_status = self._add_status_bars(screen_reduced)
        
        # Add to screen history for stacking
        self.screen_history.append(screen_with_status)
        
        # Stack the 3 most recent screens (padding with first screen if needed)
        if len(self.screen_history) < self.num_stacked_screens:
            # Pad with first screen until we have enough history
            stacked_screens = np.stack([self.screen_history[0]] * self.num_stacked_screens, axis=-1)
        else:
            stacked_screens = np.stack(list(self.screen_history), axis=-1)
        
        return stacked_screens
    
    def _add_status_bars(self, screen):
        """
        Add visual status bars as described in the proven methodology.
        Overlays bars showing HP, levels, exploration progress, and REWARD.
        """
        # Create copy to modify
        screen_with_status = screen.copy()
        
        # Get game state for status bars
        total_levels = self._get_total_pokemon_levels()
        exploration_progress = self.screen_record.get_exploration_progress()
        unique_screens = exploration_progress['unique_screens']
        
        # Get current episode reward for reward bar
        current_episode_reward = self.episode_rewards.get('total', 0)
        
        # Add status bars in top-right area (8 pixels wide, clearly visible)
        bar_x_start = 32
        bar_width = 8
        
        # 1. HP bar (based on Pokemon levels) - WHITE
        hp_percentage = min(100, total_levels * 10) / 100  # Rough HP indicator from levels
        hp_bar_width = int(bar_width * hp_percentage)
        if hp_bar_width > 0:
            screen_with_status[2:4, bar_x_start:bar_x_start+hp_bar_width] = 255  # White HP bar
        
        # 2. Level progress bar - LIGHT GRAY  
        level_bar_width = min(bar_width, total_levels)  # Up to 8 pixels for levels
        if level_bar_width > 0:
            screen_with_status[5:7, bar_x_start:bar_x_start+level_bar_width] = 200  # Gray level bar
        
        # 3. Exploration progress bar (logarithmic scale) - MEDIUM GRAY
        exploration_bar_width = min(bar_width, int(np.log10(max(1, unique_screens))))  # Log scale
        if exploration_bar_width > 0:
            screen_with_status[8:10, bar_x_start:bar_x_start+exploration_bar_width] = 150  # Darker exploration bar
        
        # 4. **NEW: REWARD BAR** - Shows current episode reward - BRIGHT (like in video)
        # Scale reward to bar width (adjust scale as needed)
        reward_scale = max(1, abs(current_episode_reward) / 100)  # Scale down large rewards
        reward_bar_width = min(bar_width, int(abs(current_episode_reward) / reward_scale))
        if reward_bar_width > 0:
            # Color based on reward: bright if positive, dim if negative
            reward_color = 220 if current_episode_reward >= 0 else 80
            screen_with_status[11:13, bar_x_start:bar_x_start+reward_bar_width] = reward_color
        
        return screen_with_status
    
    def _get_screen_hash(self):
        """Get hash of current screen to detect if visuals changed."""
        # Get screen as bytes and hash it
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen_array = np.asarray(screen_image)
        return hash(screen_array.tobytes())
    
    def _read_memory(self, address):
        """Read single byte from Game Boy memory."""
        return self.pyboy.memory[address]
    
    def _save_cnn_debug_visualization(self, observation, step_count, action_taken=None, reward=None):
        """
        Save visualization of exactly what the CNN sees.
        
        This creates a debug image showing:
        - The 3 stacked grayscale frames side by side
        - Status bars overlaid
        - Step information and game state
        
        Args:
            observation: The processed observation that goes to the CNN (36x40x3)
            step_count: Current step number
            action_taken: Action that was taken (optional)
            reward: Reward received (optional)
        """
        if not self.debug_cnn_input:
            return
            
        # Only save every N steps to avoid too many files
        if step_count % self.cnn_save_frequency != 0:
            return
            
        try:
            # Extract the 3 stacked frames
            frame1 = observation[:, :, 0]  # Oldest frame
            frame2 = observation[:, :, 1]  # Middle frame  
            frame3 = observation[:, :, 2]  # Most recent frame (newest)
            
            # Create a combined visualization - Stack frames VERTICALLY (top to bottom)
            # Top = newest (most recent), Bottom = oldest
            combined_width = 40
            combined_height = 36 * 3 + 2 * 5  # 3 frames + padding
            combined_image = np.zeros((combined_height, combined_width), dtype=np.uint8)
            
            # Place frames vertically with padding - NEWEST at TOP
            combined_image[0:36, :] = frame3      # Top: Current (newest)
            combined_image[41:77, :] = frame2     # Middle: Frame -1  
            combined_image[82:118, :] = frame1    # Bottom: Frame -2 (oldest)
            
            # Convert to RGB for text overlay
            combined_rgb = np.stack([combined_image] * 3, axis=-1)
            
            # Scale up for better visibility (4x scale)
            from PIL import Image, ImageDraw, ImageFont
            pil_image = Image.fromarray(combined_rgb).resize((combined_width * 4, combined_height * 4), Image.NEAREST)
            
            # Add text overlay with debug information
            draw = ImageDraw.Draw(pil_image)
            
            # Try to use a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Add labels and information - positioned for vertical layout
            draw.text((5, 5), f"Step: {step_count}", fill=(255, 255, 255), font=font)
            draw.text((5, 20), "Current", fill=(255, 255, 0), font=font)      # Top frame
            draw.text((5, 175), "Frame -1", fill=(255, 255, 0), font=font)    # Middle frame
            draw.text((5, 340), "Frame -2", fill=(255, 255, 0), font=font)    # Bottom frame
            
            if action_taken is not None:
                action_names = ['DOWN', 'LEFT', 'RIGHT', 'UP', 'A', 'B', 'START', 'SELECT', 'NOOP']
                action_name = action_names[action_taken] if action_taken < len(action_names) else f"ACT_{action_taken}"
                draw.text((5, 385), f"Action: {action_name}", fill=(0, 255, 0), font=font)
            
            if reward is not None:
                draw.text((5, 400), f"Reward: {reward:.3f}", fill=(255, 0, 255), font=font)
            
            # Add game state information
            total_levels = self._get_total_pokemon_levels()
            unique_screens = self.screen_record.get_exploration_progress()['unique_screens']
            draw.text((5, 415), f"Levels: {total_levels}", fill=(0, 255, 255), font=font)
            draw.text((5, 430), f"Unique Screens: {unique_screens}", fill=(0, 255, 255), font=font)
            
            # Save the debug image
            debug_filename = f"cnn_input_{step_count:06d}.png"
            debug_path = self.cnn_debug_dir / debug_filename
            pil_image.save(debug_path)
            
            # Also save the raw observation data for analysis
            raw_filename = f"cnn_data_{step_count:06d}.npy"
            raw_path = self.cnn_debug_dir / raw_filename
            np.save(raw_path, observation)
            
        except Exception as e:
            print(f"Warning: Failed to save CNN debug visualization: {e}")
    
    def _create_cnn_debug_video(self, max_frames=1000):
        """
        Create a video from saved CNN debug frames to see the model's perception over time.
        
        Args:
            max_frames: Maximum number of frames to include in video
        """
        try:
            import cv2
            
            # Find all CNN debug images
            debug_images = sorted(list(self.cnn_debug_dir.glob("cnn_input_*.png")))
            
            if not debug_images:
                print("No CNN debug images found to create video")
                return
                
            # Limit to max_frames
            if len(debug_images) > max_frames:
                debug_images = debug_images[-max_frames:]  # Take most recent frames
            
            # Read first image to get dimensions
            first_img = cv2.imread(str(debug_images[0]))
            height, width, layers = first_img.shape
            
            # Create video writer
            video_path = self.cnn_debug_dir / "cnn_perception_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
            
            print(f"Creating CNN perception video from {len(debug_images)} frames...")
            
            for img_path in debug_images:
                img = cv2.imread(str(img_path))
                video_writer.write(img)
            
            video_writer.release()
            print(f"CNN perception video saved to: {video_path}")
            
        except ImportError:
            print("OpenCV not available - install with: pip install opencv-python")
        except Exception as e:
            print(f"Failed to create CNN debug video: {e}")
    
    def _read_uint16(self, address):
        """Read 16-bit unsigned integer (little-endian)."""
        low = self._read_memory(address)
        high = self._read_memory(address + 1)
        return low | (high << 8)
    
    def _read_uint16_be(self, address):
        """Read 16-bit unsigned integer (big-endian) - used for Pokemon HP/stats."""
        high = self._read_memory(address)
        low = self._read_memory(address + 1)
        return (high << 8) | low
    
    def _read_uint24_be(self, address):
        """Read 24-bit unsigned integer (big-endian) - used for Pokemon experience."""
        byte1 = self._read_memory(address)
        byte2 = self._read_memory(address + 1)
        byte3 = self._read_memory(address + 2)
        return (byte1 << 16) | (byte2 << 8) | byte3
    
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
    
    # ============================================================================
    # SIMPLIFIED POKEMON LEVEL TRACKING
    # ============================================================================
    
    def _get_party_count(self):
        """Get number of Pokemon in party (0-6)."""
        return self._read_memory(PARTY_COUNT_ADDRESS)
    
    def _get_total_pokemon_levels(self):
        """
        Get combined levels of all Pokemon in party.
        
        This is the core progression signal from the successful implementation.
        """
        total_levels = 0
        party_count = self._read_memory(PARTY_COUNT_ADDRESS)
        
        # Read levels from each Pokemon slot
        level_addresses = [
            PARTY_POKEMON_1_LEVEL,
            PARTY_POKEMON_2_LEVEL, 
            PARTY_POKEMON_3_LEVEL,
            PARTY_POKEMON_4_LEVEL,
            PARTY_POKEMON_5_LEVEL,
            PARTY_POKEMON_6_LEVEL
        ]
        
        for i in range(min(party_count, 6)):
            level = self._read_memory(level_addresses[i])
            total_levels += level
            
        return total_levels
    
    # ============================================================================
    # GAME STATE DETECTION FUNCTIONS
    # ============================================================================
    
    def _is_in_battle(self):
        """Check if currently in a battle."""
        battle_type = self._read_memory(BATTLE_TYPE_ADDRESS)
        return battle_type != 0
    
    def _is_character_created(self):
        """Check if character creation is complete."""
        # Character creation is complete when player has a name and at least 1 Pokemon
        party_count = self._get_party_count()
        
        # Check if player name is set (not all zeros)
        player_name = [self._read_memory(PLAYER_NAME_ADDRESS + i) for i in range(7)]
        has_name = any(byte != 0 for byte in player_name)
        
        return has_name and party_count > 0
    
    def _calculate_reward(self, screen_array):
        """
        COMPLETE 7-REWARD SYSTEM FROM SUCCESSFUL IMPLEMENTATION
        
        Based on the exact reward structure that defeated Brock:
        1. 'event': self.update_max_event_rew() - Max event progress
        2. 'level': self.get_levels_reward() - Pokemon level increases  
        3. 'heal': self.total_healing_rew - Healing progress
        4. 'op_lvl': self.update_max_op_level() - Opponent level increases
        5. 'dead': -0.1*self.died_count - Death penalty
        6. 'badge': self.get_badges() * 2 - Badge progress
        7. 'explore': self.get_knn_reward() - Exploration (screen novelty)
        
        Additional rewards tested but not used:
        - party_xp: 0.1*sum(poke_xps) - Experience points
        - op_poke: self.max_opponent_poke * 800 - Opponent Pokemon
        - money: money * 3 - In-game money
        - seen_poke: seen_poke_count * 400 - Pokemon seen
        """
        
        # Initialize state_scores dictionary matching the successful implementation
        state_scores = {}
        
        # ====================================================================
        # 1. EVENT REWARD: Max event progress
        # ====================================================================
        event_reward = self._update_max_event_reward()
        state_scores['event'] = event_reward
        
        # ====================================================================  
        # 2. LEVEL REWARD: Pokemon level increases
        # ====================================================================
        level_reward = self._get_levels_reward()
        state_scores['level'] = level_reward
        
        # ====================================================================
        # 3. HEAL REWARD: Total healing progress
        # ====================================================================
        heal_reward = self._get_total_healing_reward()
        state_scores['heal'] = heal_reward
        
        # ====================================================================
        # 4. OPPONENT LEVEL REWARD: Max opponent level increases
        # ====================================================================
        op_lvl_reward = self._update_max_op_level_reward()
        state_scores['op_lvl'] = op_lvl_reward
        
        # ====================================================================
        # 5. DEATH PENALTY: -0.1 * died_count
        # ====================================================================
        death_penalty = -0.1 * self.died_count
        state_scores['dead'] = death_penalty
        
        # ====================================================================
        # 6. BADGE REWARD: badges * 2
        # ====================================================================
        badge_reward = self._get_badges_reward() * 2
        state_scores['badge'] = badge_reward
        
        # ====================================================================
        # 7. EXPLORATION REWARD: Screen novelty (KNN-based)
        # ====================================================================
        explore_reward = self._get_knn_reward(screen_array)
        state_scores['explore'] = explore_reward
        
        # ====================================================================
        # TOTAL REWARD CALCULATION
        # ====================================================================
        total_reward = sum(state_scores.values())
        
        # Update episode tracking
        for key, value in state_scores.items():
            if key not in self.episode_rewards:
                self.episode_rewards[key] = 0
            self.episode_rewards[key] += value
        
        self.episode_rewards['total'] += total_reward
        
        # Debug output for significant rewards
        significant_rewards = {k: v for k, v in state_scores.items() if abs(v) > 0.01}
        if significant_rewards:
            # Only print when multiple rewards are active or very significant single rewards
            if len(significant_rewards) > 1 or any(abs(v) > 1.0 for v in significant_rewards.values()):
                reward_str = ", ".join([f"{k}:{v:.2f}" for k, v in significant_rewards.items()])
                print(f"  Multi-reward: {reward_str}")
        
        return total_reward
    
    # ============================================================================
    # INDIVIDUAL REWARD FUNCTIONS (7-REWARD SYSTEM)
    # ============================================================================
    
    def _update_max_event_reward(self):
        """Event progress reward - tracks game progression milestones."""
        # Simple event tracking based on game state progression
        current_map = self._read_memory(MAP_ID_ADDRESS)
        party_count = self._read_memory(PARTY_COUNT_ADDRESS)
        
        # Calculate event score based on game progression
        event_score = 0
        if party_count > 0:  # Got starter Pokemon
            event_score += 1
        if current_map > 0:  # Left starting area
            event_score += 1
        if current_map >= 50:  # Advanced map progression
            event_score += 1
            
        # Only reward increases
        if not hasattr(self, 'max_event_score'):
            self.max_event_score = 0
        
        if event_score > self.max_event_score:
            reward = event_score - self.max_event_score
            self.max_event_score = event_score
            return reward
        return 0
    
    def _get_levels_reward(self):
        """Pokemon level progression reward."""
        current_total_levels = self._get_total_pokemon_levels()
        
        # Only reward level increases
        if current_total_levels > self.previous_total_levels:
            reward = current_total_levels - self.previous_total_levels
            self.previous_total_levels = current_total_levels
            return reward
        return 0
    
    def _get_total_healing_reward(self):
        """Healing progress reward - encourages using Pokemon Centers."""
        # Track total healing by monitoring HP restoration
        if not hasattr(self, 'total_healing_rew'):
            self.total_healing_rew = 0
            self.last_party_hp = 0
        
        # Get current party HP
        current_hp = 0
        party_count = self._read_memory(PARTY_COUNT_ADDRESS)
        
        # Simple HP tracking (would need proper HP addresses for full implementation)
        # For now, estimate based on party count and levels
        total_levels = self._get_total_pokemon_levels()
        estimated_max_hp = total_levels * 20  # Rough estimate
        estimated_current_hp = estimated_max_hp  # Assume full HP for simplicity
        
        # Check for healing (HP increase) - return only the incremental healing
        healing_reward = 0
        if estimated_current_hp > self.last_party_hp:
            healing = estimated_current_hp - self.last_party_hp
            healing_reward = healing * 0.01  # Small healing reward
            self.total_healing_rew += healing_reward  # Track cumulative for info
        
        self.last_party_hp = estimated_current_hp
        return healing_reward  # 🔧 FIXED: Return only this step's reward, not cumulative
    
    def _update_max_op_level_reward(self):
        """Opponent level reward - encourages battling stronger opponents."""
        if not hasattr(self, 'max_op_level'):
            self.max_op_level = 0
        
        # In battle, try to estimate opponent level
        # This would need proper battle memory addresses for full implementation
        is_in_battle = self._is_in_battle()
        if is_in_battle:
            # Estimate opponent level based on own level (rough approximation)
            own_levels = self._get_total_pokemon_levels()
            estimated_op_level = max(1, own_levels // 6)  # Rough estimate
            
            if estimated_op_level > self.max_op_level:
                reward = estimated_op_level - self.max_op_level
                self.max_op_level = estimated_op_level
                return reward
        return 0
    
    def _get_badges_reward(self):
        """Badge progression reward."""
        # Count badge bits
        badge_flags = self._read_memory(BADGE_FLAGS_ADDRESS)
        badge_count = bin(badge_flags).count('1')
        
        # Only reward new badges
        if not hasattr(self, 'previous_badge_count'):
            self.previous_badge_count = 0
        
        if badge_count > self.previous_badge_count:
            reward = badge_count - self.previous_badge_count
            self.previous_badge_count = badge_count
            return reward
        return 0
    
    def _get_knn_reward(self, screen_array):
        """Exploration reward using screen novelty (KNN-based)."""
        # This is our existing screen novelty detection
        is_novel = self.screen_record.is_novel_screen(screen_array)
        
        if is_novel:
            # Consistent with original exploration reward
            return 1.0
        return 0
    
    def _track_deaths(self):
        """Track Pokemon deaths for penalty calculation."""
        # Simple death tracking - could be enhanced with actual HP monitoring
        if not hasattr(self, 'last_total_levels'):
            self.last_total_levels = self._get_total_pokemon_levels()
            return
        
        current_levels = self._get_total_pokemon_levels()
        
        # If levels suddenly drop significantly, might indicate death/PC boxing
        if current_levels < self.last_total_levels - 5:  # Threshold for significant drop
            potential_deaths = (self.last_total_levels - current_levels) // 5
            self.died_count += potential_deaths
        
        self.last_total_levels = current_levels
    
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
        Simplified to match proven methodology - focus on key metrics.
        """
        # Get exploration progress from screen record
        exploration_progress = self.screen_record.get_exploration_progress()
        
        # Get current Pokemon levels for progression tracking
        current_total_levels = self._get_total_pokemon_levels()
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_rewards['total'],
            'reward_breakdown': self.episode_rewards,
            
            # Core progression metrics (simplified)
            'unique_screens': exploration_progress['unique_screens'],
            'total_pokemon_levels': current_total_levels,
            'current_position': self._get_current_position(),
            
            # Basic game state
            'map_id': self._read_memory(MAP_ID_ADDRESS),
            'party_count': self._read_memory(PARTY_COUNT_ADDRESS),
        }
        
        return info
    
    def _get_current_position(self):
        """Get current player position for monitoring."""
        x = self._read_memory(PLAYER_X_ADDRESS)
        y = self._read_memory(PLAYER_Y_ADDRESS)
        map_id = self._read_memory(MAP_ID_ADDRESS)
        return {'x': x, 'y': y, 'map': map_id}
    
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
    
    def _capture_screenshot(self):
        """
        Capture periodic screenshot for monitoring.
        """
        # Get current screen (PyBoy 2.0 API) - convert to RGB
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen = np.asarray(screen_image)
        
        # Save screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_step_{self.current_step}_{timestamp}.png"
        filepath = self.screenshots_dir / filename
        
        img = Image.fromarray(screen)
        img.save(filepath)
        
        progress = self.screen_record.get_exploration_progress()
        print(f"\n📸 Screenshot: {filename}")
        print(f"   Step: {self.current_step}, Unique screens: {progress['unique_screens']}")
    
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
        # Use shared session folder with env-specific subfolders for screenshots only
        env_config = config.copy()
        # Don't create separate env folders - use the main session folder
        # Each env will create its own screenshot subfolder if needed
        env_config['env_rank'] = rank  # Pass rank for screenshot organization
        
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

# Create session directory with timestamp - this will contain all env folders
SESSION_NAME = f"pokemon_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
SESSION_PATH = Path('./sessions') / SESSION_NAME

print(f"\nSession Directory: {SESSION_PATH}")

# Environment configuration
ENV_CONFIG = {
    'rom_path': ROM_PATH,
    'headless': True,  # No visualization during training (faster)
    'action_freq': 24,  # Execute 24 frames per action
    'max_steps': 8192,  # Maximum steps per episode
    'save_path': str(SESSION_PATH),  # Each env will create subfolders under this
    'save_screenshots': True,
    'debug_cnn_input': True,  # 🔬 Enable CNN input debugging visualization
    'cnn_save_frequency': 100,  # Save CNN debug frames every 50 steps
}

def main(continue_training=False, model_path=None):
    """Main training function with support for continuing from existing models."""
    # Training hyperparameters - BALANCED LEARNING SETTINGS
    NUM_ENVS = min(3, os.cpu_count())  # Parallel environments
    TOTAL_TIMESTEPS = 1_000_000  # Learn game progression
    SAVE_FREQ = 5_000  # Save checkpoint every 5k steps
    LEARNING_RATE = 0.0001
    N_STEPS = 512  # ⬆️ INCREASED from 64 - More stable updates every 256*4=1024 steps
    BATCH_SIZE = 512  # Match N_STEPS for efficient learning
    N_EPOCHS = 8  # ⬇️ REDUCED from 8 - Prevent overfitting to random exploration
    GAMMA = 0.9995  # ⬆️ INCREASED from 0.9 - Long-term strategy important for Pokemon
    GAE_LAMBDA = 0.995  # ⬆️ INCREASED from 0.85 - Better advantage estimation

    print(f"\nTraining Configuration:")
    print(f"  Parallel Environments: {NUM_ENVS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Steps per Update: {N_STEPS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gamma (discount): {GAMMA}")
    print(f"  GAE Lambda: {GAE_LAMBDA}")
    
    if continue_training and model_path:
        print(f"  🔄 Continuing from: {model_path}")
    else:
        print(f"  🆕 Starting fresh training")

    # ============================================================================
    # MAIN TRAINING LOOP
    # ============================================================================

    print(f"\nStarting Training...")
    print("=" * 80)

    # Create parallel environments
    print(f"Creating {NUM_ENVS} parallel environments...")
    env = SubprocVecEnv([make_env(i, ENV_CONFIG) for i in range(NUM_ENVS)])

    if continue_training and model_path and Path(model_path).exists():
        print(f"Loading existing model from: {model_path}")
        model = PPO.load(model_path, env=env)
        print("✅ Model loaded successfully! Continuing training...")
    else:
        print("Initializing new PPO model...")
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
            clip_range=0.02,  # ⬇️ REDUCED from 0.4 - more conservative policy updates
            clip_range_vf=None, # No value function clipping
            ent_coef=0.35,  # ⬆️ INCREASED from 0.15 - start higher to prevent collapse
            # This allows early exploration but doesn't overwhelm game rewards
            # 0.25 is moderate-high - enough exploration without entropy addiction
            # Will be reduced over time via callback to allow strategy development
            vf_coef=0.5,  # ⬆️ INCREASED from 0.3 - value function helps learn game strategy
            max_grad_norm=0.5, # ⬇️ REDUCED from 0.8 - more stable training
            use_sde=False, # 🔧 FIXED: Disable SDE for discrete actions (Pokemon uses discrete action space)
            sde_sample_freq=-1, # No SDE
            target_kl=None,  # Remove KL limit - allow reasonable policy changes
            tensorboard_log=str(SESSION_PATH / 'tensorboard'),
            policy_kwargs=dict(
                features_extractor_kwargs=dict(features_dim=512)
            ),
            verbose=1,
            seed=42,  # for reproducibility, set a specific seed here
            device='auto' # use GPU if available (CUDA)
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
        json.dump(milestone_callback.training_history, f, indent=2)
    print(f"Training history: {history_path}")

    # Close environments
    env.close()

    print("\nTraining session complete!")
    print(f"View results with: tensorboard --logdir {SESSION_PATH / 'tensorboard'}")

    # ============================================================================
    # AUTOMATIC DEMO AFTER TRAINING
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("LAUNCHING POST-TRAINING DEMO")
    print("=" * 80)
    print("Starting demo of trained model...")
    print("Press Ctrl+C to stop the demo early")
    
    # Launch demo automatically
    demo_model(final_model_path, show_visual=True, max_minutes=5)

def demo_model(model_path, show_visual=True, max_minutes=None):
    """
    Live demo of the trained model playing Pokemon Red.
    
    This is a pure demonstration mode - no training, no step limits,
    just watch the AI play the game naturally until you stop it.
    
    Args:
        model_path: Path to the trained model
        show_visual: Whether to show the game window (should be True for demos)
        max_minutes: Optional time limit in minutes (None = no limit)
    """
    print(f"\n🎮 LIVE DEMO MODE - Watch the AI play Pokemon Red!")
    print(f"Model: {model_path}")
    print(f"Visual: {'ON' if show_visual else 'OFF'}")
    if max_minutes:
        print(f"Time Limit: {max_minutes} minutes")
    else:
        print(f"Time Limit: None (press Ctrl+C to stop)")
    print("-" * 60)
    
    if not show_visual:
        print("⚠️  WARNING: Visual demo with show_visual=False doesn't make sense!")
        print("   Enabling visuals for proper demo experience...")
        show_visual = True
    
    # Create demo environment - ALWAYS with visuals for demo
    demo_config = {
        'rom_path': ROM_PATH,
        'headless': False,  # Always show window for demos
        'force_visual': True,  # Override Windows headless forcing
        'action_freq': 24,
        'max_steps': 999999,  # Very high limit, essentially no limit
        'save_path': './demo_temp',
        'save_screenshots': False,
        'session_path': './demo_temp'  # Separate demo session
    }
    
    try:
        # Create single environment (no parallel for demo)
        env = PokemonRedEnv(demo_config)
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Verify PyBoy window state
        if hasattr(env.pyboy, 'window_type'):
            print(f"🖼️  PyBoy window type: {env.pyboy.window_type}")
        else:
            print(f"🖼️  PyBoy window type: {'Visual' if not env.headless else 'Headless'}")
        
        if not env.headless:
            print("🎮 PyBoy visual window should now be visible!")
            print("   If you don't see it, check your taskbar or try Alt+Tab")
            print("   The window title should be 'PyBoy'")
            
            # Try to bring window to front (Windows-specific)
            try:
                import platform
                if platform.system() == 'Windows':
                    import time
                    time.sleep(0.5)  # Let window initialize
                    print("   Attempting to bring PyBoy window to front...")
                    # This requires pywin32, but let's try a basic approach
                    import subprocess
                    subprocess.run(['powershell', '-Command', 
                                  'Add-Type -AssemblyName Microsoft.VisualBasic; ' +
                                  '[Microsoft.VisualBasic.Interaction]::AppActivate("PyBoy")'], 
                                  capture_output=True)
            except:
                pass  # Window focus is optional
        else:
            print("⚠️  Warning: Running in headless mode (no visual window)")
        
        # Reset environment
        obs, info = env.reset()
        
        # Demo statistics (just for interest)
        total_reward = 0
        action_counts = {i: 0 for i in range(9)}
        action_names = ['DOWN', 'LEFT', 'RIGHT', 'UP', 'A', 'B', 'START', 'SELECT', 'WAIT']
        
        print(f"\n🚀 Starting live demo...")
        print(f"   Watch the PyBoy game window!")
        print(f"   Stats will be shown every 500 actions")
        print(f"   Press Ctrl+C to stop the demo")
        print("-" * 60)
        
        import time
        start_time = time.time()
        step = 0
        
        # Run demo - INFINITE LOOP until user stops
        while True:
            # Check time limit if specified
            if max_minutes:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= max_minutes:
                    print(f"\n⏰ Time limit reached ({max_minutes} minutes)")
                    break
            
            # Get action from trained model - use some randomness for natural behavior
            action, _states = model.predict(obs, deterministic=False)
            action = int(action)
            
            # Track action for stats
            action_counts[action] += 1
            step += 1
            
            # Take step in the game
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Print stats every 500 steps (not too frequently)
            if step % 500 == 0:
                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                
                # Calculate action distribution
                total_actions = sum(action_counts.values())
                top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                action_dist = " | ".join([f"{action_names[act]}:{count*100/total_actions:.1f}%" 
                                        for act, count in top_actions])
                
                print(f"⏱️  {minutes:02d}:{seconds:02d} | Actions: {step:5d} | Top: {action_dist}")
            
            # If episode ends naturally, just continue (don't reset)
            # This lets the AI deal with game over states naturally
            if terminated or truncated:
                print(f"🔄 Episode transition at step {step} (continuing...)")
                obs, info = env.reset()
            
            # Comfortable viewing pace - not too fast
            time.sleep(0.1)  # 10 FPS for easy watching
        
        # Final statistics
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n" + "=" * 60)
        print(f"DEMO SUMMARY")
        print(f"" + "=" * 60)
        print(f"Duration: {minutes:02d}:{seconds:02d}")
        print(f"Total Actions: {step}")
        print(f"Actions per Minute: {step/(elapsed_time/60):.1f}")
        
        print(f"\nAction Distribution:")
        total_actions = sum(action_counts.values())
        for i, name in enumerate(action_names):
            count = action_counts[i]
            pct = (count / total_actions * 100) if total_actions > 0 else 0
            if pct > 1:  # Only show actions used more than 1%
                bar_length = int(pct / 2)
                bar = "█" * bar_length
                print(f"  {name:6s}: {pct:5.1f}% {bar}")
        
        # Close environment cleanly
        env.close()
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo stopped by user")
        print(f"   Total actions taken: {step if 'step' in locals() else 0}")
        try:
            env.close()
        except:
            pass
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\nDemo session ended! 🎉")


# ============================================================================
# MAIN EXECUTION WITH MENU SYSTEM
# ============================================================================
if __name__ == '__main__':
    # This is required for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    
    # Interactive menu system
    def show_menu():
        """Display interactive menu for training or demo."""
        print("\n" + "=" * 80)
        print("🎮 POKEMON RED RL - TRAINING & DEMO SYSTEM")
        print("=" * 80)
        
        # Check for existing models
        sessions_dir = Path('./sessions')
        available_models = []
        
        if sessions_dir.exists():
            for session_folder in sessions_dir.iterdir():
                if session_folder.is_dir():
                    final_model = session_folder / 'final_model.zip'
                    best_model = session_folder / 'checkpoints' / 'best_model.zip'
                    
                    if final_model.exists():
                        available_models.append({
                            'name': f"{session_folder.name} (final)",
                            'path': final_model,
                            'session': session_folder.name
                        })
                    
                    if best_model.exists():
                        available_models.append({
                            'name': f"{session_folder.name} (best)",
                            'path': best_model,
                            'session': session_folder.name
                        })
        
        print(f"Available options:")
        print(f"  1. 🏋️  Start New Training Session")
        print(f"  2. 🔄  Continue Training from Checkpoint")
        print(f"  3. 🎯  Statistics Demo (no visuals - action stats)")
        print(f"  4. 🎮  Live Visual Demo (watch AI play Pokemon Red)")
        
        if available_models:
            print(f"\nAvailable trained models:")
            for i, model in enumerate(available_models):
                print(f"  {i+5}. 🎮 Live Demo: {model['name']}")
        else:
            print(f"\n  No trained models found in ./sessions/")
        
        print(f"\n  0. ❌ Exit")
        print("=" * 80)
        
        while True:
            try:
                choice = input("\nEnter your choice: ").strip()
                
                if choice == '0':
                    print("Goodbye! 👋")
                    return
                
                elif choice == '1':
                    print(f"\n🏋️  Starting new training session...")
                    main()
                    return
                
                elif choice == '3':
                    if not available_models:
                        print("❌ No trained models available for demo!")
                        continue
                    
                    print(f"\n🎯 Select model for statistics demo (no visuals):")
                    for i, model in enumerate(available_models):
                        print(f"  {i+1}. {model['name']}")
                    
                    model_choice = input("Enter model number: ").strip()
                    try:
                        model_idx = int(model_choice) - 1
                        if 0 <= model_idx < len(available_models):
                            selected_model = available_models[model_idx]
                            print(f"\n🎯 Running statistics demo with {selected_model['name']}")
                            demo_model(selected_model['path'], show_visual=False, max_minutes=3)
                        else:
                            print("❌ Invalid model selection!")
                    except ValueError:
                        print("❌ Invalid input!")
                    return
                
                elif choice == '4':
                    # Continue training from checkpoint
                    print(f"\n🔄 Continue training options:")
                    print(f"  1. Continue from latest checkpoint")
                    print(f"  2. Select specific checkpoint")
                    
                    continue_choice = input("Enter choice (1-2): ").strip()
                    
                    if continue_choice == '1':
                        # Auto-find latest checkpoint
                        checkpoint_dir = SESSION_PATH / 'checkpoints'
                        if checkpoint_dir.exists():
                            checkpoints = list(checkpoint_dir.glob('ppo_pokemon_*.zip'))
                            if checkpoints:
                                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                                print(f"🔄 Continuing from latest: {latest.name}")
                                main(continue_training=True, model_path=str(latest))
                            else:
                                print("❌ No checkpoints found! Starting fresh training instead.")
                                main()
                        else:
                            print("❌ No checkpoints found! Starting fresh training instead.")
                            main()
                    elif continue_choice == '2':
                        # List all available checkpoints
                        all_checkpoints = []
                        sessions_dir = Path('./sessions')
                        if sessions_dir.exists():
                            for session_dir in sessions_dir.iterdir():
                                if session_dir.is_dir():
                                    checkpoint_dir = session_dir / 'checkpoints'
                                    if checkpoint_dir.exists():
                                        for checkpoint in checkpoint_dir.glob('ppo_pokemon_*.zip'):
                                            all_checkpoints.append({
                                                'path': str(checkpoint),
                                                'name': f"{session_dir.name}/{checkpoint.name}",
                                                'mtime': checkpoint.stat().st_mtime
                                            })
                        
                        if not all_checkpoints:
                            print("❌ No checkpoints found! Starting fresh training instead.")
                            main()
                            return
                        
                        # Sort by modification time (newest first)
                        all_checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
                        
                        print(f"\nAvailable checkpoints:")
                        for i, checkpoint in enumerate(all_checkpoints[:10]):  # Show latest 10
                            print(f"  {i+1}. {checkpoint['name']}")
                        
                        checkpoint_choice = input("Enter checkpoint number: ").strip()
                        try:
                            checkpoint_idx = int(checkpoint_choice) - 1
                            if 0 <= checkpoint_idx < len(all_checkpoints):
                                selected_checkpoint = all_checkpoints[checkpoint_idx]
                                print(f"🔄 Continuing from: {selected_checkpoint['name']}")
                                main(continue_training=True, model_path=selected_checkpoint['path'])
                            else:
                                print("❌ Invalid checkpoint selection!")
                        except ValueError:
                            print("❌ Invalid input!")
                    else:
                        print("❌ Invalid choice!")
                    return
                
                elif choice == '3':
                    if not available_models:
                        print("❌ No trained models available for demo!")
                        continue
                    
                    print(f"\n🎯 Select model for statistics demo (no visuals):")
                    for i, model in enumerate(available_models):
                        print(f"  {i+1}. {model['name']}")
                    
                    model_choice = input("Enter model number: ").strip()
                    try:
                        model_idx = int(model_choice) - 1
                        if 0 <= model_idx < len(available_models):
                            selected_model = available_models[model_idx]
                            print(f"\n🎯 Running statistics demo with {selected_model['name']}")
                            demo_model(selected_model['path'], show_visual=False, max_minutes=3)
                        else:
                            print("❌ Invalid model selection!")
                    except ValueError:
                        print("❌ Invalid input!")
                    return
                
                elif choice == '4':
                    if not available_models:
                        print("❌ No trained models available for demo!")
                        continue
                    
                    print(f"\n🎮 Select model for live visual demo:")
                    for i, model in enumerate(available_models):
                        print(f"  {i+1}. {model['name']}")
                    
                    model_choice = input("Enter model number: ").strip()
                    try:
                        model_idx = int(model_choice) - 1
                        if 0 <= model_idx < len(available_models):
                            selected_model = available_models[model_idx]
                            print(f"\n🎮 Running live visual demo with {selected_model['name']}")
                            demo_model(selected_model['path'], show_visual=True, max_minutes=None)
                        else:
                            print("❌ Invalid model selection!")
                    except ValueError:
                        print("❌ Invalid input!")
                    return
                
                elif choice.isdigit() and int(choice) >= 5:
                    model_idx = int(choice) - 5
                    if model_idx < len(available_models):
                        selected_model = available_models[model_idx]
                        print(f"\n🎮 Running demo with {selected_model['name']}")
                        demo_model(selected_model['path'], show_visual=True, max_minutes=None)
                        return
                    else:
                        print("❌ Invalid selection!")
                
                else:
                    print("❌ Invalid choice! Please try again.")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                return
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Launch menu by default, but allow direct training and continuing
    if len(sys.argv) > 1:
        if sys.argv[1] == '--train':
            # Direct training mode
            print("🏋️  Starting direct training mode...")
            main()
        elif sys.argv[1] == '--continue':
            # Continue training mode
            continue_training = True
            model_path = None
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
            else:
                # Default to latest checkpoint in current session
                checkpoint_dir = SESSION_PATH / 'checkpoints'
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob('ppo_pokemon_*.zip'))
                    if checkpoints:
                        # Sort by modification time, get latest
                        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                        model_path = str(latest)
                        print(f"Found latest checkpoint: {model_path}")
                    else:
                        print("No checkpoints found in current session. Starting fresh training.")
                        continue_training = False
            
            print("🔄 Starting continued training mode...")
            main(continue_training=continue_training, model_path=model_path)
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("\nPokemon Red PPO Training")
            print("Usage:")
            print("  python pokemon_red_training.py                    # Interactive menu")
            print("  python pokemon_red_training.py --train            # Start fresh training")
            print("  python pokemon_red_training.py --continue         # Continue from latest checkpoint")
            print("  python pokemon_red_training.py --continue [path]  # Continue from specific model")
            print("\nExample:")
            print("  python pokemon_red_training.py --continue sessions/session_20241220_143022/checkpoints/ppo_pokemon_50000.zip")
            print("")
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information.")
            sys.exit(1)
    else:
        # Interactive menu mode
        show_menu()

# ============================================================================
# HELPER FUNCTIONS FOR TRAINING AND EVALUATION
# ============================================================================

def create_ppo_model(env, total_timesteps=1000000):
    """
    Create PPO model with proven hyperparameters for Pokemon Red.
    
    Based on successful implementation that defeated Brock.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Wrap environment if needed
    if not hasattr(env, 'num_envs'):
        env = DummyVecEnv([lambda: env])
    
    # PPO hyperparameters optimized for screen-based exploration
    model = PPO(
        'CnnPolicy',  # CNN policy for image observations
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device='auto'
    )
    
    return model

def save_training_results(results, save_path):
    """Save training results to JSON file."""
    import json
    from pathlib import Path
    
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = save_dir / 'training_results.json'
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Training results saved to: {results_file}")

def load_training_results(save_path):
    """Load training results from JSON file."""
    import json
    from pathlib import Path
    
    results_file = Path(save_path) / 'training_results.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        return None