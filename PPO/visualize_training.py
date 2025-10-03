"""
Pokemon Red RL - Training Visualization and Analysis

This script generates comprehensive visualizations of training progress including:
- Reward curves over time
- Episode length progression
- Exploration metrics
- Badge acquisition timeline
- Performance heatmaps
- Comparative analysis across multiple runs

Usage:
    python visualize_training.py --session path/to/session/dir
    python visualize_training.py --compare session1 session2 session3
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("POKEMON RED RL - TRAINING VISUALIZATION")
print("=" * 80)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def load_training_history(session_path):
    """
    Load training history from session directory.
    
    Args:
        session_path: Path to training session directory
        
    Returns:
        Dictionary containing training history
    """
    session_path = Path(session_path)
    
    # Try to load training_history.json
    history_file = session_path / 'training_history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        print(f"✓ Loaded training history: {history_file}")
        return history
    
    # Try to load from TensorBoard logs (alternative)
    tb_path = session_path / 'tensorboard'
    if tb_path.exists():
        print(f"✓ Found TensorBoard logs: {tb_path}")
        print("  Note: Use 'tensorboard --logdir {tb_path}' for interactive visualization")
        return None
    
    print(f"❌ No training history found in {session_path}")
    return None

def plot_reward_curve(history, save_path=None):
    """
    Plot reward progression over training.
    
    Creates a line plot showing mean episode reward over timesteps,
    with a smoothed trend line for clarity.
    """
    print("\n📈 Generating reward curve...")
    
    timesteps = history['timesteps']
    rewards = history['mean_reward']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw rewards
    ax.plot(timesteps, rewards, alpha=0.3, color='blue', label='Raw Reward')
    
    # Plot smoothed rewards (moving average)
    window = min(50, len(rewards) // 10)
    if window > 1:
        smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
        ax.plot(timesteps, smoothed, linewidth=2, color='darkblue', label=f'Smoothed (window={window})')
    
    # Formatting
    ax.set_xlabel('Training Timesteps', fontsize=12)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_title('Training Reward Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    max_reward = max(rewards)
    final_reward = rewards[-1]
    ax.text(0.02, 0.98, 
            f'Max Reward: {max_reward:.2f}\nFinal Reward: {final_reward:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

def plot_episode_length(history, save_path=None):
    """
    Plot episode length progression.
    
    Shows how long episodes last over training. Longer episodes generally
    indicate the agent is surviving and progressing further.
    """
    print("\n📏 Generating episode length plot...")
    
    timesteps = history['timesteps']
    lengths = history['episode_length']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot lengths
    ax.plot(timesteps, lengths, alpha=0.4, color='green')
    
    # Smoothed version
    window = min(50, len(lengths) // 10)
    if window > 1:
        smoothed = pd.Series(lengths).rolling(window=window, min_periods=1).mean()
        ax.plot(timesteps, smoothed, linewidth=2, color='darkgreen', label='Smoothed')
    
    # Formatting
    ax.set_xlabel('Training Timesteps', fontsize=12)
    ax.set_ylabel('Episode Length (steps)', fontsize=12)
    ax.set_title('Episode Length Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics
    mean_length = np.mean(lengths)
    max_length = max(lengths)
    ax.axhline(mean_length, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_length:.0f}')
    ax.text(0.02, 0.98,
            f'Mean Length: {mean_length:.0f}\nMax Length: {max_length:.0f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

def plot_exploration_metrics(history, save_path=None):
    """
    Plot exploration metrics over time.
    
    Shows how many unique locations the agent has discovered.
    """
    print("\n🗺️ Generating exploration plot...")
    
    if 'exploration' not in history or not history['exploration']:
        print("  ⚠️ No exploration data available")
        return None
    
    timesteps = history['timesteps']
    exploration = history['exploration']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(timesteps, exploration, linewidth=2, color='purple')
    ax.fill_between(timesteps, exploration, alpha=0.3, color='purple')
    
    ax.set_xlabel('Training Timesteps', fontsize=12)
    ax.set_ylabel('Unique Coordinates Explored', fontsize=12)
    ax.set_title('Exploration Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Statistics
    final_exploration = exploration[-1] if exploration else 0
    ax.text(0.02, 0.98,
            f'Total Coordinates: {final_exploration:.0f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

def plot_badge_progression(history, save_path=None):
    """
    Plot gym badge acquisition over time.
    
    Shows cumulative badges obtained, which is a key progress metric.
    """
    print("\n🏅 Generating badge progression plot...")
    
    if 'badges' not in history or not history['badges']:
        print("  ⚠️ No badge data available")
        return None
    
    timesteps = history['timesteps']
    badges = history['badges']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Step plot for discrete badges
    ax.step(timesteps, badges, where='post', linewidth=2, color='gold')
    ax.fill_between(timesteps, badges, alpha=0.3, color='gold', step='post')
    
    # Reference lines for each badge
    for i in range(1, 9):
        ax.axhline(i, color='gray', linestyle='--', alpha=0.2)
        ax.text(timesteps[-1], i, f'Badge {i}', 
                verticalalignment='center', fontsize=9)
    
    ax.set_xlabel('Training Timesteps', fontsize=12)
    ax.set_ylabel('Gym Badges Obtained', fontsize=12)
    ax.set_title('Badge Acquisition Timeline', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.5, 8.5)
    ax.grid(True, alpha=0.3)
    
    # Final count
    final_badges = badges[-1] if badges else 0
    ax.text(0.02, 0.98,
            f'Final Badges: {final_badges:.0f} / 8',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

def plot_pokemon_caught(history, save_path=None):
    """
    Plot Pokemon caught over time.
    """
    print("\n⚡ Generating Pokemon caught plot...")
    
    if 'pokemon_caught' not in history or not history['pokemon_caught']:
        print("  ⚠️ No Pokemon caught data available")
        return None
    
    timesteps = history['timesteps']
    pokemon = history['pokemon_caught']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(timesteps, pokemon, linewidth=2, color='red', marker='o', markersize=3)
    ax.fill_between(timesteps, pokemon, alpha=0.3, color='red')
    
    ax.set_xlabel('Training Timesteps', fontsize=12)
    ax.set_ylabel('Pokemon Caught', fontsize=12)
    ax.set_title('Pokemon Collection Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Final count
    final_pokemon = pokemon[-1] if pokemon else 0
    ax.text(0.02, 0.98,
            f'Total Caught: {final_pokemon:.0f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

def create_summary_dashboard(history, save_path=None):
    """
    Create comprehensive dashboard with all metrics.
    
    Shows 6 subplots in a 2x3 grid for complete overview.
    """
    print("\n📊 Generating comprehensive dashboard...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    timesteps = history['timesteps']
    
    # 1. Reward curve
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = history['mean_reward']
    ax1.plot(timesteps, rewards, alpha=0.3, color='blue')
    if len(rewards) > 10:
        smoothed = pd.Series(rewards).rolling(window=min(50, len(rewards)//10), min_periods=1).mean()
        ax1.plot(timesteps, smoothed, linewidth=2, color='darkblue')
    ax1.set_title('Mean Episode Reward', fontweight='bold')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode length
    ax2 = fig.add_subplot(gs[0, 1])
    lengths = history['episode_length']
    ax2.plot(timesteps, lengths, alpha=0.4, color='green')
    if len(lengths) > 10:
        smoothed = pd.Series(lengths).rolling(window=min(50, len(lengths)//10), min_periods=1).mean()
        ax2.plot(timesteps, smoothed, linewidth=2, color='darkgreen')
    ax2.set_title('Episode Length', fontweight='bold')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Steps')
    ax2.grid(True, alpha=0.3)
    
    # 3. Exploration
    ax3 = fig.add_subplot(gs[1, 0])
    if 'exploration' in history and history['exploration']:
        ax3.plot(timesteps, history['exploration'], linewidth=2, color='purple')
        ax3.fill_between(timesteps, history['exploration'], alpha=0.3, color='purple')
    ax3.set_title('Exploration Progress', fontweight='bold')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Coordinates')
    ax3.grid(True, alpha=0.3)
    
    # 4. Badges
    ax4 = fig.add_subplot(gs[1, 1])
    if 'badges' in history and history['badges']:
        ax4.step(timesteps, history['badges'], where='post', linewidth=2, color='gold')
        ax4.fill_between(timesteps, history['badges'], alpha=0.3, color='gold', step='post')
    ax4.set_title('Gym Badges Obtained', fontweight='bold')
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Badges')
    ax4.set_ylim(-0.5, 8.5)
    ax4.grid(True, alpha=0.3)
    
    # 5. Pokemon caught
    ax5 = fig.add_subplot(gs[2, 0])
    if 'pokemon_caught' in history and history['pokemon_caught']:
        ax5.plot(timesteps, history['pokemon_caught'], linewidth=2, color='red')
        ax5.fill_between(timesteps, history['pokemon_caught'], alpha=0.3, color='red')
    ax5.set_title('Pokemon Caught', fontweight='bold')
    ax5.set_xlabel('Timesteps')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # Calculate statistics
    stats_text = "Training Summary\n" + "="*30 + "\n\n"
    stats_text += f"Total Timesteps: {timesteps[-1]:,}\n"
    stats_text += f"Mean Reward: {np.mean(rewards):.2f}\n"
    stats_text += f"Max Reward: {max(rewards):.2f}\n"
    stats_text += f"Final Reward: {rewards[-1]:.2f}\n\n"
    stats_text += f"Mean Episode Length: {np.mean(lengths):.0f}\n"
    stats_text += f"Max Episode Length: {max(lengths):.0f}\n\n"
    
    if 'badges' in history and history['badges']:
        final_badges = history['badges'][-1]
        stats_text += f"Final Badges: {final_badges:.0f} / 8\n"
    
    if 'exploration' in history and history['exploration']:
        final_exploration = history['exploration'][-1]
        stats_text += f"Coordinates Explored: {final_exploration:.0f}\n"
    
    if 'pokemon_caught' in history and history['pokemon_caught']:
        final_pokemon = history['pokemon_caught'][-1]
        stats_text += f"Pokemon Caught: {final_pokemon:.0f}\n"
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Main title
    fig.suptitle('Pokemon Red RL Training Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

def compare_multiple_runs(session_paths, save_path=None):
    """
    Compare multiple training runs side-by-side.
    
    Useful for comparing different hyperparameters or architectures.
    """
    print(f"\n🔬 Comparing {len(session_paths)} training runs...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(session_paths)))
    
    for idx, (session_path, color) in enumerate(zip(session_paths, colors)):
        history = load_training_history(session_path)
        if not history:
            continue
        
        label = Path(session_path).name
        timesteps = history['timesteps']
        
        # Plot rewards
        rewards = history['mean_reward']
        smoothed = pd.Series(rewards).rolling(window=min(50, len(rewards)//10), min_periods=1).mean()
        axes[0, 0].plot(timesteps, smoothed, linewidth=2, color=color, label=label, alpha=0.7)
        
        # Plot episode lengths
        lengths = history['episode_length']
        smoothed = pd.Series(lengths).rolling(window=min(50, len(lengths)//10), min_periods=1).mean()
        axes[0, 1].plot(timesteps, smoothed, linewidth=2, color=color, label=label, alpha=0.7)
        
        # Plot badges
        if 'badges' in history and history['badges']:
            axes[1, 0].step(timesteps, history['badges'], where='post', 
                           linewidth=2, color=color, label=label, alpha=0.7)
        
        # Plot exploration
        if 'exploration' in history and history['exploration']:
            axes[1, 1].plot(timesteps, history['exploration'], 
                           linewidth=2, color=color, label=label, alpha=0.7)
    
    # Format plots
    axes[0, 0].set_title('Mean Episode Reward', fontweight='bold')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Episode Length', fontweight='bold')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Gym Badges', fontweight='bold')
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Badges')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Exploration', fontweight='bold')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Coordinates')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize Pokemon Red RL Training')
    parser.add_argument('--session', type=str, help='Path to training session directory')
    parser.add_argument('--compare', nargs='+', help='Multiple session paths to compare')
    parser.add_argument('--output', type=str, default='./visualizations',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Single session visualization
    if args.session:
        session_path = Path(args.session)
        print(f"\n📂 Loading session: {session_path}")
        
        history = load_training_history(session_path)
        
        if history:
            # Generate all plots
            plot_reward_curve(history, output_dir / 'reward_curve.png')
            plot_episode_length(history, output_dir / 'episode_length.png')
            plot_exploration_metrics(history, output_dir / 'exploration.png')
            plot_badge_progression(history, output_dir / 'badges.png')
            plot_pokemon_caught(history, output_dir / 'pokemon_caught.png')
            create_summary_dashboard(history, output_dir / 'dashboard.png')
            
            print("\n✅ All visualizations generated!")
            print(f"📁 Saved to: {output_dir}")
        
        plt.show()
    
    # Compare multiple sessions
    elif args.compare:
        print(f"\n📊 Comparing {len(args.compare)} sessions...")
        compare_multiple_runs(args.compare, output_dir / 'comparison.png')
        
        print("\n✅ Comparison visualization generated!")
        print(f"📁 Saved to: {output_dir}")
        
        plt.show()
    
    else:
        print("\n❌ Please specify --session or --compare")
        parser.print_help()

if __name__ == '__main__':
    main()