#!/usr/bin/env python3
"""
Direct Pokemon Red Training Script with Continuation Support
============================================================

This script provides a simple way to start or continue Pokemon Red RL training
without going through the interactive menu system.

Usage:
    python train_pokemon.py                     # Start fresh training
    python train_pokemon.py --continue          # Continue from latest checkpoint
    python train_pokemon.py --continue [path]   # Continue from specific model
    python train_pokemon.py --help             # Show help

Author: GitHub Copilot
Date: December 2024
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

def main():
    """Main training function with direct execution."""
    try:
        # Import the main training function
        from pokemon_red_training import main as training_main, SESSION_PATH
        
        print("🚀 Pokemon Red Direct Training Script")
        print("=" * 50)
        
        # Check for command line arguments
        continue_training = False
        model_path = None
        
        if len(sys.argv) > 1:
            if sys.argv[1] == "--continue":
                continue_training = True
                if len(sys.argv) > 2:
                    model_path = sys.argv[2]
                    if not Path(model_path).exists():
                        print(f"❌ Error: Model file not found: {model_path}")
                        return
                else:
                    # Auto-find latest checkpoint
                    checkpoint_dir = SESSION_PATH / 'checkpoints'
                    if checkpoint_dir.exists():
                        checkpoints = list(checkpoint_dir.glob('ppo_pokemon_*.zip'))
                        if checkpoints:
                            # Sort by modification time, get latest
                            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                            model_path = str(latest)
                            print(f"📁 Found latest checkpoint: {latest.name}")
                        else:
                            print("⚠️  No checkpoints found in current session.")
                            print("   Starting fresh training instead...")
                            continue_training = False
                    else:
                        print("⚠️  No checkpoint directory found.")
                        print("   Starting fresh training instead...")
                        continue_training = False
            
            elif sys.argv[1] in ["--help", "-h"]:
                print_help()
                return
            
            else:
                print(f"❌ Error: Unknown argument '{sys.argv[1]}'")
                print("   Use --help for usage information.")
                return
        
        # Display training mode
        if continue_training and model_path:
            print(f"🔄 Mode: Continue Training")
            print(f"📁 Model: {model_path}")
        else:
            print(f"🆕 Mode: Fresh Training")
        
        print("\n� Starting Pokemon Red RL Training...")
        print("   Press Ctrl+C to stop training")
        print("=" * 50)
        
        # Start training
        training_main(continue_training=continue_training, model_path=model_path)
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        print("   Progress has been saved to checkpoints.")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure you're in the correct directory with pokemon_red_training.py")
        
    except Exception as e:
        print(f"❌ Training Error: {e}")
        print("   Check the error details above for troubleshooting.")


def print_help():
    """Print help information."""
    print("""
Pokemon Red Direct Training Script
==================================

This script allows you to start or continue Pokemon Red RL training
without using the interactive menu system.

Usage:
  python train_pokemon.py                     # Start fresh training
  python train_pokemon.py --continue          # Continue from latest checkpoint  
  python train_pokemon.py --continue [path]   # Continue from specific model
  python train_pokemon.py --help             # Show this help

Examples:
  # Start new training session
  python train_pokemon.py
  
  # Continue from automatically-found latest checkpoint
  python train_pokemon.py --continue
  
  # Continue from specific checkpoint
  python train_pokemon.py --continue sessions/session_20241220_143022/checkpoints/ppo_pokemon_50000.zip

Features:
  🔄 Automatic checkpoint detection and loading
  📊 Full TensorBoard logging and monitoring  
  🎮 CNN input debugging and visualization
  🏗️  Parallel environment training (6 environments)
  ⚡ Optimized hyperparameters for Pokemon Red
  
Training Configuration:
  • Total Timesteps: 1,000,000
  • Checkpoint Frequency: Every 5,000 steps
  • Learning Rate: 0.0001 (adaptive)
  • Parallel Environments: 6
  • Reward System: 7-component (event, level, heal, op_lvl, dead, badge, explore)
  
Monitoring:
  • TensorBoard: tensorboard --logdir sessions/[session]/tensorboard
  • CNN Debug: Check session folder for visualization images
  • Progress: Watch terminal output and reward trends

For more advanced options, use the main interactive script:
  python pokemon_red_training.py
""")


def start_training():
    """Legacy function for backwards compatibility."""
    main()


if __name__ == "__main__":
    main()