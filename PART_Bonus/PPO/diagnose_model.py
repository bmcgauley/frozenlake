"""
Pokemon Red RL - Diagnostic Tool

This script loads a trained model and provides detailed diagnostics about
what the agent is doing, what it's seeing, and what rewards it's receiving.

Usage:
    python diagnose_model.py --model path/to/model.zip --steps 1000
"""

import os
import argparse
import numpy as np
import time
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Import training components
from stable_baselines3 import PPO
from pokemon_red_training import PokemonRedEnv
from screen_state_tracker import ScreenStateTracker

# Action names for better logging
ACTION_NAMES = ['DOWN', 'LEFT', 'RIGHT', 'UP', 'A', 'B', 'START', 'SELECT', 'WAIT']


def diagnose_model(model_path, num_steps=1000, save_screens=True, output_dir='./diagnosis'):
    """
    Run diagnostic on a trained model.
    
    Args:
        model_path: Path to the trained model .zip file
        num_steps: Number of steps to run
        save_screens: Whether to save screen captures
        output_dir: Directory to save outputs
    """
    print("=" * 80)
    print("POKEMON RED RL - MODEL DIAGNOSTICS")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Steps: {num_steps}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    screens_path = output_path / 'screens'
    screens_path.mkdir(exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create environment
    print("\nCreating environment...")
    config = {
        'rom_path': 'PokemonRed.gb',
        'headless': True,
        'action_freq': 24,
        'max_steps': num_steps,
        'save_path': str(output_path),
        'save_screenshots': False,
    }
    
    try:
        env = PokemonRedEnv(config)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return
    
    # Initialize tracking
    action_history = []
    reward_history = []
    screen_hashes = []
    position_history = []
    
    reward_breakdown = {
        'exploration': [],
        'battle_won': [],
        'battle_engaged': [],
        'damage_dealt': [],
        'pokemon_caught': [],
        'gym_badge': [],
        'death_penalty': [],
        'stuck_penalty': [],
        'menu_penalty': [],
        'total': []
    }
    
    # Independent screen tracker for analysis
    screen_tracker = ScreenStateTracker(history_size=100, short_term_size=20)
    
    # Reset environment
    print("\nStarting diagnostic run...")
    obs, info = env.reset()
    
    total_reward = 0
    step = 0
    
    # Save initial screen
    if save_screens:
        screen_img = Image.fromarray(obs)
        screen_img.save(screens_path / f'screen_{step:05d}.png')
    
    print("\nStep | Action  | Reward  | Cum.Rew | Stagnation | Position")
    print("-" * 80)
    
    for step in range(num_steps):
        # Get action from model
        action, _states = model.predict(obs, deterministic=False)
        action = int(action)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track data
        action_history.append(action)
        reward_history.append(reward)
        total_reward += reward
        
        # Track reward breakdown
        for key in reward_breakdown:
            reward_breakdown[key].append(env.episode_rewards.get(key, 0))
        
        # Get screen from environment for analysis
        screen_image = env.pyboy.screen.image.convert('RGB')
        screen_full = np.asarray(screen_image)
        
        # Analyze with independent tracker
        screen_analysis = screen_tracker.update(screen_full)
        screen_hashes.append(screen_analysis['screen_hash'])
        
        # Track position
        pos_x = env._read_memory(env.PLAYER_X_ADDRESS) if hasattr(env, 'PLAYER_X_ADDRESS') else 0
        pos_y = env._read_memory(env.PLAYER_Y_ADDRESS) if hasattr(env, 'PLAYER_Y_ADDRESS') else 0
        map_id = env._read_memory(env.MAP_ID_ADDRESS) if hasattr(env, 'MAP_ID_ADDRESS') else 0
        position_history.append((map_id, pos_x, pos_y))
        
        # Print progress every 10 steps
        if step % 10 == 0:
            stagnation_info = ""
            if screen_analysis['is_stuck']:
                stagnation_info = f"STUCK({screen_analysis['consecutive_same']})"
            elif screen_analysis['is_loop']:
                stagnation_info = f"LOOP({screen_analysis['loop_length']})"
            elif screen_analysis['is_duplicate']:
                stagnation_info = f"DUP({screen_analysis['duplicate_count']})"
            else:
                stagnation_info = "NEW"
            
            print(f"{step:4d} | {ACTION_NAMES[action]:7s} | {reward:+7.2f} | {total_reward:+7.1f} | {stagnation_info:14s} | ({map_id:3d},{pos_x:3d},{pos_y:3d})")
        
        # Save screen periodically
        if save_screens and step % 50 == 0:
            screen_img = Image.fromarray(obs)
            screen_img.save(screens_path / f'screen_{step:05d}.png')
        
        # Check termination
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    # Action distribution
    print("\nAction Distribution:")
    action_counter = Counter(action_history)
    for action_idx in range(9):
        count = action_counter.get(action_idx, 0)
        percentage = (count / len(action_history)) * 100 if action_history else 0
        print(f"  {ACTION_NAMES[action_idx]:7s}: {count:5d} ({percentage:5.1f}%)")
    
    # Screen diversity
    unique_screens = len(set(screen_hashes))
    diversity_ratio = unique_screens / len(screen_hashes) if screen_hashes else 0
    print(f"\nScreen Diversity:")
    print(f"  Total screens: {len(screen_hashes)}")
    print(f"  Unique screens: {unique_screens}")
    print(f"  Diversity ratio: {diversity_ratio:.2%}")
    
    # Position tracking
    unique_positions = len(set(position_history))
    print(f"\nPosition Exploration:")
    print(f"  Unique positions: {unique_positions}")
    
    # Most common positions (might indicate stuck spots)
    position_counter = Counter(position_history)
    print(f"  Most common positions:")
    for pos, count in position_counter.most_common(5):
        percentage = (count / len(position_history)) * 100
        print(f"    Map {pos[0]:3d} at ({pos[1]:3d},{pos[2]:3d}): {count:5d} steps ({percentage:5.1f}%)")
    
    # Reward analysis
    print(f"\nReward Summary:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {total_reward/len(reward_history) if reward_history else 0:.3f}")
    print(f"\nReward Breakdown (cumulative):")
    for key, values in reward_breakdown.items():
        if values:
            final_value = values[-1]
            print(f"  {key:20s}: {final_value:+10.2f}")
    
    # Screen tracker statistics
    tracker_stats = screen_tracker.get_statistics()
    print(f"\nScreen Tracker Statistics:")
    print(f"  Loop detections: {tracker_stats['loop_detections']}")
    print(f"  Diversity ratio: {tracker_stats['diversity_ratio']:.2%}")
    
    # Create visualizations
    print(f"\nGenerating visualizations...")
    
    # Plot 1: Reward over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Cumulative reward
    axes[0].plot(np.cumsum(reward_history))
    axes[0].set_title('Cumulative Reward Over Time')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].grid(True)
    
    # Reward per step (smoothed)
    window = 50
    if len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed)
        axes[1].set_title(f'Reward per Step (smoothed, window={window})')
    else:
        axes[1].plot(reward_history)
        axes[1].set_title('Reward per Step')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Reward')
    axes[1].grid(True)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Action distribution over time
    action_windows = []
    for i in range(0, len(action_history), 50):
        window_actions = action_history[i:i+50]
        unique_actions = len(set(window_actions))
        action_windows.append(unique_actions)
    
    axes[2].plot(action_windows)
    axes[2].set_title('Action Diversity Over Time (unique actions per 50 steps)')
    axes[2].set_xlabel('Window')
    axes[2].set_ylabel('Unique Actions')
    axes[2].grid(True)
    axes[2].set_ylim(0, 9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'diagnostic_plots.png', dpi=150)
    print(f"✓ Saved plots to {output_path / 'diagnostic_plots.png'}")
    
    # Save detailed log
    log_data = {
        'model_path': model_path,
        'num_steps': step + 1,
        'total_reward': float(total_reward),
        'action_distribution': {ACTION_NAMES[i]: int(action_counter.get(i, 0)) for i in range(9)},
        'screen_diversity': {
            'total': len(screen_hashes),
            'unique': unique_screens,
            'ratio': float(diversity_ratio)
        },
        'position_exploration': {
            'unique_positions': unique_positions,
            'most_common': [(list(pos), int(count)) for pos, count in position_counter.most_common(10)]
        },
        'reward_breakdown': {k: float(v[-1]) if v else 0 for k, v in reward_breakdown.items()},
        'tracker_statistics': tracker_stats
    }
    
    with open(output_path / 'diagnosis.json', 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"✓ Saved detailed log to {output_path / 'diagnosis.json'}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    
    # Clean up
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Diagnose Pokemon Red RL model')
    parser.add_argument('--model', type=str, required=True, help='Path to model .zip file')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps to run')
    parser.add_argument('--save-screens', action='store_true', help='Save screen captures')
    parser.add_argument('--output', type=str, default='./diagnosis', help='Output directory')
    
    args = parser.parse_args()
    
    diagnose_model(
        model_path=args.model,
        num_steps=args.steps,
        save_screens=args.save_screens,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
