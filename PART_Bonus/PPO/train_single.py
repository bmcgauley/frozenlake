"""
Single Environment Pokemon Red Training
For when you want to avoid multiprocessing issues and see cleaner output.
"""

import os
import time
from pokemon_red_training import PokemonRedEnv, create_ppo_model, save_training_results
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

def train_pokemon_single():
    """
    Train Pokemon Red RL agent with single environment (no multiprocessing).
    Cleaner for debugging and monitoring progress.
    """
    
    print("=" * 80)
    print("POKEMON RED RL - SINGLE ENVIRONMENT TRAINING")
    print("=" * 80)
    print("Configuration:")
    print("  Environments: 1 (no multiprocessing)")
    print("  Timesteps: 50,000")
    print("  Methodology: Screen novelty + Level progression")
    print("  Output: Clean, minimal spam")
    print()
    
    # Create save directory
    save_path = f'./sessions/pokemon_single_{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_path, exist_ok=True)
    
    # Create single environment
    config = {
        'rom_path': 'PokemonRed.gb',
        'headless': True,
        'max_steps': 8192,
        'save_path': save_path,
        'save_screenshots': False,  # No screenshots to reduce spam
        'action_freq': 24
    }
    
    env = DummyVecEnv([lambda: PokemonRedEnv(config)])
    print("✓ Single environment created")
    
    # Create model
    model = create_ppo_model(env, total_timesteps=50000)
    print("✓ PPO model ready")
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"{save_path}/checkpoints/",
        name_prefix="pokemon_single"
    )
    
    print("\n🚀 Starting training...")
    print("Progress updates every 2048 steps")
    print("Novel screen updates every 50 discoveries")
    print()
    
    start_time = time.time()
    
    try:
        # Train
        model.learn(
            total_timesteps=50000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        # Save final model
        final_model_path = f"{save_path}/final_model.zip"
        model.save(final_model_path)
        
        print("\n" + "=" * 80)
        print("✓ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Time: {training_time/60:.1f} minutes")
        print(f"Model: {final_model_path}")
        
        # Save results
        results = {
            'timesteps': 50000,
            'training_time_minutes': training_time / 60,
            'environments': 1,
            'methodology': 'screen_novelty_plus_levels',
            'final_model_path': final_model_path,
            'status': 'completed'
        }
        
        save_training_results(results, save_path)
        print(f"Results: {save_path}/training_results.json")
        
        env.close()
        
        return final_model_path
        
    except KeyboardInterrupt:
        print("\n⏹️  Training stopped by user")
        
        # Save interrupted model
        interrupt_path = f"{save_path}/interrupted_model.zip"
        model.save(interrupt_path)
        print(f"Saved: {interrupt_path}")
        
        env.close()
        return interrupt_path
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        env.close()
        return None

if __name__ == "__main__":
    train_pokemon_single()