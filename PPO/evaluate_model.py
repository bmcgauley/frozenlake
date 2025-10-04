"""
Pokemon Red RL - Model Evaluation and Demo Script

This script loads a trained model and evaluates its performance.
It can run in two modes:
1. Visual demo mode with rendering
2. Evaluation mode to collect statistics

Usage:
    python evaluate_model.py --model path/to/model.zip --mode demo
    python evaluate_model.py --model path/to/model.zip --mode eval --episodes 10
"""

import os
import argparse
import numpy as np
import time
from pathlib import Path
import json
from datetime import datetime
from PIL import Image

# Import training components
from stable_baselines3 import PPO
from pyboy import PyBoy
import gymnasium as gym
from gymnasium import spaces

# ============================================================================
# EVALUATION ENVIRONMENT (Same as training but with rendering)
# ============================================================================

class PokemonRedEvalEnv(gym.Env):
    """
    Evaluation environment for Pokemon Red.
    Similar to training environment but optimized for visualization.
    """
    
    def __init__(self, rom_path, headless=False, save_screenshots=True, screenshot_dir='./eval_screenshots'):
        super(PokemonRedEvalEnv, self).__init__()
        
        self.rom_path = rom_path
        self.headless = headless
        self.save_screenshots = save_screenshots
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PyBoy (PyBoy 2.0 API)
        window_type = 'null' if headless else 'SDL2'
        self.pyboy = PyBoy(
            self.rom_path,
            window=window_type
        )
        
        if not headless:
            self.pyboy.set_emulation_speed(1)  # Normal speed for viewing
        else:
            self.pyboy.set_emulation_speed(0)  # Max speed
        
        # Get screen interface (PyBoy 2.0 API)
        self.screen = self.pyboy.screen
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(120, 128, 3), dtype=np.uint8
        )
        
        # Action mapping (PyBoy 2.0 uses string button names)
        self.action_map = [
            'down', 'left', 'right', 'up',
            'a', 'b', 'start', 'select', None  # None for no-op
        ]
        
        # Tracking variables
        self.current_step = 0
        self.episode_reward = 0
        self.milestones = []
        self.visited_coordinates = set()
        
        # Memory addresses (same as training)
        self.PLAYER_X = 0xD362
        self.PLAYER_Y = 0xD361
        self.MAP_ID = 0xD35E
        self.BADGE_FLAGS = 0xD356
        self.PARTY_SIZE = 0xD163
        
        print(f"Evaluation environment initialized")
        print(f"Headless: {headless}")
        print(f"Screenshot directory: {self.screenshot_dir}")
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Start from ROM boot (no save state)
        # Run a few frames to stabilize
        for _ in range(60):
            self.pyboy.tick()
        
        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0
        self.visited_coordinates = set()
        self.milestones = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute action."""
        # Log the action being taken
        action_name = self.action_map[action] if action < len(self.action_map) else 'none'
        if action_name is None:
            action_name = 'wait'
        
        # Execute action
        self._take_action(action)
        
        # Get observation
        obs = self._get_observation()
        
        # Simple reward (just for tracking)
        reward = self._simple_reward()
        self.episode_reward += reward
        
        # Update step
        self.current_step += 1
        
        # Print action every 10 steps to avoid spam
        if self.current_step % 10 == 0:
            pos = self._get_position()
            print(f"Step {self.current_step:5d} | Action: {action_name:6s} | Pos: ({pos[0]:3d}, {pos[1]:3d}, Map:{pos[2]:3d}) | Reward: {reward:+.1f} | Total: {self.episode_reward:+.1f}")
        
        # Check termination
        done = self.current_step >= 10000
        
        return obs, reward, done, False, self._get_info()
    
    def _take_action(self, action):
        """Execute action in emulator."""
        button = self.action_map[action]
        
        # Press button (PyBoy 2.0 API)
        if button is not None:
            self.pyboy.button_press(button)
        
        for _ in range(8):
            self.pyboy.tick()
        
        # Release button (PyBoy 2.0 API)
        if button is not None:
            self.pyboy.button_release(button)
        
        for _ in range(16):
            self.pyboy.tick()
    
    def _get_observation(self):
        """Get screen observation."""
        # PyBoy 2.0 API: Convert PIL Image to RGB numpy array
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen = np.asarray(screen_image)
        
        # Downsample
        from skimage.transform import resize
        downsampled = resize(
            screen, (120, 128),
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.uint8)
        
        return downsampled
    
    def _get_position(self):
        """Get current player position."""
        x = self.pyboy.memory[self.PLAYER_X]
        y = self.pyboy.memory[self.PLAYER_Y]
        m = self.pyboy.memory[self.MAP_ID]
        return (x, y, m)
    
    def _simple_reward(self):
        """Calculate simple reward for evaluation."""
        reward = 0
        
        # Exploration
        x, y, m = self._get_position()
        pos = (m, x, y)
        
        if pos not in self.visited_coordinates:
            self.visited_coordinates.add(pos)
            reward += 1.0
        
        return reward
    
    def _get_info(self):
        """Get info dict."""
        badges = bin(self.pyboy.memory[self.BADGE_FLAGS]).count('1')
        
        return {
            'step': self.current_step,
            'reward': self.episode_reward,
            'badges': badges,
            'coordinates_explored': len(self.visited_coordinates)
        }
    
    def capture_screenshot(self, prefix='screenshot'):
        """Capture and save current screen."""
        # PyBoy 2.0 API: Convert PIL Image to RGB
        screen_image = self.pyboy.screen.image.convert('RGB')
        screen = np.asarray(screen_image)
        img = Image.fromarray(screen)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.png"
        filepath = self.screenshot_dir / filename
        
        img.save(filepath)
        print(f"Screenshot saved: {filepath}")
        
        return filepath
    
    def render(self):
        """Render current state."""
        return self._get_observation()
    
    def close(self):
        """Close emulator."""
        self.pyboy.stop()

# ============================================================================
# DEMO MODE - Visual demonstration with human interaction
# ============================================================================

def run_demo(model_path, rom_path, num_steps=10000):
    """
    Run visual demonstration of trained model.
    
    Args:
        model_path: Path to trained model .zip file
        rom_path: Path to Pokemon Red ROM
        num_steps: Number of steps to run
    """
    print("\n" + "=" * 80)
    print("DEMO MODE - Visual Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Steps: {num_steps}")
    print("\nControls:")
    print("  ESC - Exit demo")
    print("  SPACE - Take screenshot")
    print("  P - Pause/Resume")
    print("=" * 80)
    
    # Create environment with rendering
    env = PokemonRedEvalEnv(rom_path, headless=False)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded successfully")
    
    # Reset environment
    obs, info = env.reset()
    
    # Run demo
    print("\nStarting demo...")
    paused = False
    step = 0
    
    try:
        while step < num_steps:
            if not paused:
                # Get action from model
                action, _states = model.predict(obs, deterministic=True)
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                
                step += 1
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    print(f"Step {step}/{num_steps} | "
                          f"Reward: {info['reward']:.1f} | "
                          f"Badges: {info['badges']} | "
                          f"Explored: {info['coordinates_explored']}")
                
                # Auto-screenshot on badge acquisition
                if info['badges'] > 0 and step % 500 == 0:
                    env.capture_screenshot(f"badge_{info['badges']}")
                
                if done:
                    print("\n✓ Episode completed!")
                    obs, info = env.reset()
            
            # Small delay for visibility
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    
    finally:
        # Final screenshot
        env.capture_screenshot('final')
        
        # Print final stats
        print("\n" + "=" * 80)
        print("DEMO COMPLETED")
        print("=" * 80)
        print(f"Total Steps: {step}")
        print(f"Final Reward: {info['reward']:.2f}")
        print(f"Badges Obtained: {info['badges']}")
        print(f"Coordinates Explored: {info['coordinates_explored']}")
        print(f"Screenshots saved to: {env.screenshot_dir}")
        
        env.close()

# ============================================================================
# EVALUATION MODE - Statistical evaluation across multiple episodes
# ============================================================================

def run_evaluation(model_path, rom_path, num_episodes=10):
    """
    Run statistical evaluation of trained model.
    
    Args:
        model_path: Path to trained model .zip file
        rom_path: Path to Pokemon Red ROM
        num_episodes: Number of episodes to evaluate
    """
    print("\n" + "=" * 80)
    print("EVALUATION MODE - Statistical Analysis")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 80)
    
    # Create headless environment for speed
    env = PokemonRedEvalEnv(rom_path, headless=True, save_screenshots=False)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded successfully")
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    episode_badges = []
    episode_explorations = []
    
    # Run evaluation episodes
    print(f"\nRunning {num_episodes} evaluation episodes...\n")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"Episode {episode + 1}/{num_episodes}...", end=' ', flush=True)
        
        while not done and episode_length < 10000:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        # Store statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_badges.append(info['badges'])
        episode_explorations.append(info['coordinates_explored'])
        
        print(f"Reward: {episode_reward:.1f}, "
              f"Length: {episode_length}, "
              f"Badges: {info['badges']}, "
              f"Explored: {info['coordinates_explored']}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_badges = np.mean(episode_badges)
    mean_exploration = np.mean(episode_explorations)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean:   {mean_reward:.2f}")
    print(f"  Std:    {std_reward:.2f}")
    print(f"  Min:    {min(episode_rewards):.2f}")
    print(f"  Max:    {max(episode_rewards):.2f}")
    print(f"\nPerformance Metrics:")
    print(f"  Mean Episode Length:  {mean_length:.0f} steps")
    print(f"  Mean Badges Obtained: {mean_badges:.1f}")
    print(f"  Mean Exploration:     {mean_exploration:.0f} coordinates")
    
    # Save results
    results = {
        'model': str(model_path),
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_badges': episode_badges,
        'episode_explorations': episode_explorations,
        'statistics': {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_length': float(mean_length),
            'mean_badges': float(mean_badges),
            'mean_exploration': float(mean_exploration)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to file
    results_path = Path(model_path).parent / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    env.close()
    
    return results

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Pokemon Red RL Agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--rom', type=str, default='PokemonRed.gb',
                       help='Path to Pokemon Red ROM')
    parser.add_argument('--mode', type=str, choices=['demo', 'eval'], default='demo',
                       help='Evaluation mode: demo (visual) or eval (statistical)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes for eval mode')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of steps for demo mode')
    
    args = parser.parse_args()
    
    # Verify model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        exit(1)
    
    # Verify ROM exists
    if not os.path.exists(args.rom):
        print(f"ERROR: ROM not found at {args.rom}")
        exit(1)
    
    # Run appropriate mode
    if args.mode == 'demo':
        run_demo(args.model, args.rom, args.steps)
    else:
        run_evaluation(args.model, args.rom, args.episodes)
    
    print("\nEvaluation complete!")