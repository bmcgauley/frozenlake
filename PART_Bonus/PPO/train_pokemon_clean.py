"""
Clean Pokemon Red Training Script
Optimized version with reduced console spam and better error handling.
"""

import os
import numpy as np
from pokemon_red_training import PokemonRedEnv, create_ppo_model, save_training_results
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import time

def make_env(rank=0):
    """
    Create environment function for multiprocessing.
    """
    def _init():
        config = {
            'rom_path': 'PokemonRed.gb',
            'headless': True,  # Always headless for training
            'max_steps': 8192,  # Standard episode length
            'save_path': f'./sessions/pokemon_training_{time.strftime("%Y%m%d_%H%M%S")}',
            'save_screenshots': False,  # Disable screenshots during training to reduce spam
            'action_freq': 24  # Grid-aligned movement
        }
        return PokemonRedEnv(config)
    return _init

def create_training_callbacks(save_path):
    """Create callbacks for training monitoring."""
    
    # Checkpoint callback - save model every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_path}/checkpoints/",
        name_prefix="pokemon_rl"
    )
    
    return CallbackList([checkpoint_callback])

def train_pokemon_red(
    timesteps=100000,
    n_envs=4,  # Number of parallel environments
    save_freq=10000,
    verbose=True
):
    """
    Train Pokemon Red RL agent with clean output and proper error handling.
    """
    
    print("=" * 80)
    print("POKEMON RED RL TRAINING - SCREEN-BASED EXPLORATION")
    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Parallel Environments: {n_envs}")
    print(f"  Checkpoint Frequency: {save_freq:,}")
    print(f"  Methodology: Screen novelty + Level progression")
    print()
    
    # Create save directory
    save_path = f'./sessions/pokemon_training_{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Create vectorized environment
        if n_envs == 1:
            # Single environment for debugging
            env = DummyVecEnv([make_env(0)])
        else:
            # Multiple environments for faster training
            env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        
        print(f"✓ Created {n_envs} training environment(s)")
        
        # Create PPO model with optimized hyperparameters
        model = create_ppo_model(env, total_timesteps=timesteps)
        print("✓ PPO model created")
        
        # Create callbacks
        callbacks = create_training_callbacks(save_path)
        print("✓ Training callbacks configured")
        
        # Start training
        print(f"\n🚀 Starting training for {timesteps:,} timesteps...")
        print("   Progress will be shown every 2048 steps")
        print("   Novel screen discovery will be logged every 50 screens")
        print("   Level progress will be logged immediately")
        print()
        
        start_time = time.time()
        
        # Train the model
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = f"{save_path}/final_model.zip"
        model.save(final_model_path)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Training Time: {training_time/60:.1f} minutes")
        print(f"Final Model: {final_model_path}")
        print(f"Checkpoints: {save_path}/checkpoints/")
        
        # Save training results
        results = {
            'timesteps': timesteps,
            'training_time_minutes': training_time / 60,
            'n_envs': n_envs,
            'methodology': 'screen_novelty_plus_levels',
            'final_model_path': final_model_path,
            'status': 'completed'
        }
        
        save_training_results(results, save_path)
        print(f"Results: {save_path}/training_results.json")
        
        # Clean up
        env.close()
        
        return final_model_path, save_path
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        print("Saving current progress...")
        
        try:
            # Save current model
            interrupt_model_path = f"{save_path}/interrupted_model.zip"
            model.save(interrupt_model_path)
            print(f"Model saved: {interrupt_model_path}")
            
            # Save results
            results = {
                'timesteps': 'interrupted',
                'methodology': 'screen_novelty_plus_levels',
                'model_path': interrupt_model_path,
                'status': 'interrupted'
            }
            save_training_results(results, save_path)
            
        except:
            print("Failed to save interrupted model")
        
        env.close()
        return None, save_path
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("Check the error above and try again")
        
        try:
            env.close()
        except:
            pass
            
        return None, save_path

if __name__ == "__main__":
    # Default training configuration
    train_pokemon_red(
        timesteps=100000,  # 100k timesteps
        n_envs=4,          # 4 parallel environments
        save_freq=10000,   # Save every 10k steps
        verbose=True
    )