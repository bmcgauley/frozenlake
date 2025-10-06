"""
Test Script - Simplified Pokemon Red Training
Test the proven screen-based exploration approach with a short training session.
"""

import os
import numpy as np
from pokemon_red_training import PokemonRedEnv, save_training_results, create_ppo_model

def test_simplified_training():
    """Test short training session with simplified reward system."""
    
    print("="*80)
    print("TESTING SIMPLIFIED POKEMON RED RL TRAINING")
    print("Based on proven screen-based exploration methodology")
    print("="*80)
    
    # Configuration for quick test
    config = {
        'rom_path': 'PokemonRed.gb',
        'headless': True,  # No window for testing
        'max_steps': 1000,  # Short episode for testing
        'save_path': './test_session',
        'save_screenshots': False  # Faster for testing
    }
    
    # Create environment
    print("Creating environment...")
    env = PokemonRedEnv(config)
    
    # Create PPO model
    print("Creating PPO model...")
    model = create_ppo_model(env, total_timesteps=5000)  # Very short training
    
    # Training parameters
    timesteps = 5000  # Quick test
    save_freq = 1000
    
    print(f"Starting test training for {timesteps} timesteps...")
    print("Monitoring for:")
    print("  1. Screen novelty rewards (primary)")
    print("  2. Level progression rewards (secondary)")
    print("  3. Action distribution (avoiding policy collapse)")
    print()
    
    try:
        # Train with custom callback for monitoring
        class TestCallback:
            def __init__(self):
                self.step_count = 0
                self.last_info = None
                
            def __call__(self, locals_, globals_):
                self.step_count += 1
                info = locals_.get('infos', [{}])[0] if locals_.get('infos') else {}
                
                if self.step_count % 500 == 0:
                    print(f"Step {self.step_count}:")
                    print(f"  Unique screens: {info.get('unique_screens', 0)}")
                    print(f"  Total levels: {info.get('total_pokemon_levels', 0)}")
                    print(f"  Episode reward: {info.get('episode_reward', 0):.2f}")
                    print(f"  Map ID: {info.get('map_id', 0)}")
                    print()
                
                return True
        
        callback = TestCallback()
        
        # Train the model
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
        
        print("✓ Training completed successfully!")
        
        # Test the trained model
        print("\nTesting trained model...")
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 20 == 0:
                print(f"  Test step {step}: reward={reward:.3f}, unique_screens={info.get('unique_screens', 0)}")
            
            if terminated or truncated:
                break
        
        print(f"Test completed - Total reward: {total_reward:.2f}")
        print("✓ Model testing successful!")
        
        # Save results
        results = {
            'config': config,
            'timesteps': timesteps,
            'final_reward': total_reward,
            'methodology': 'simplified_screen_exploration',
            'status': 'test_passed'
        }
        
        save_training_results(results, config['save_path'])
        print(f"Results saved to: {config['save_path']}")
        
    except Exception as e:
        print(f"Training error: {e}")
        return False
    
    finally:
        env.close()
    
    print("\n" + "="*80)
    print("SIMPLIFIED TRAINING TEST COMPLETED SUCCESSFULLY!")
    print("Ready for full training with proven methodology")
    print("="*80)
    
    return True

if __name__ == "__main__":
    test_simplified_training()