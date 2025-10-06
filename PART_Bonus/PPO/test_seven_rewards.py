"""
Test the Complete 7-Reward System
Tests the exact reward structure from the successful Pokemon Red implementation.
"""

import os
import time
from pokemon_red_training import PokemonRedEnv, create_ppo_model, save_training_results
from stable_baselines3.common.vec_env import DummyVecEnv

def test_seven_reward_system():
    """Test the complete 7-reward system."""
    
    print("=" * 80)
    print("TESTING COMPLETE 7-REWARD SYSTEM")
    print("=" * 80)
    print("Reward Components (from successful implementation):")
    print("  1. 'event': Event progress rewards")
    print("  2. 'level': Pokemon level increases") 
    print("  3. 'heal': Healing progress")
    print("  4. 'op_lvl': Opponent level increases")
    print("  5. 'dead': Death penalty (-0.1 * died_count)")
    print("  6. 'badge': Badge progress (* 2)")
    print("  7. 'explore': Screen novelty (KNN-based)")
    print()
    
    # Create save directory
    save_path = f'./sessions/seven_rewards_{time.strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(save_path, exist_ok=True)
    
    # Create environment
    config = {
        'rom_path': 'PokemonRed.gb',
        'headless': True,
        'max_steps': 2048,
        'save_path': save_path,
        'save_screenshots': False
    }
    
    env = DummyVecEnv([lambda: PokemonRedEnv(config)])
    print(f"✓ Environment created")
    
    # Create model
    model = create_ppo_model(env, total_timesteps=5000)
    print("✓ CNN model created")
    
    print("\n🚀 Testing 7-reward system for 5k timesteps...")
    print("Monitoring all reward components:")
    print()
    
    start_time = time.time()
    
    try:
        # Training with reward monitoring
        model.learn(
            total_timesteps=5000,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print("\n✓ Training completed successfully!")
        print(f"Time: {training_time:.1f} seconds")
        
        # Test the model and analyze rewards
        print("\nTesting reward system...")
        obs = env.reset()
        
        # Get environment instance to check reward breakdown
        env_instance = env.envs[0]
        
        print(f"\nInitial reward state:")
        for key, value in env_instance.episode_rewards.items():
            print(f"  {key}: {value:.3f}")
        
        # Run some steps to see reward distribution
        for step in range(20):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if step % 5 == 0:
                print(f"\nStep {step} - Reward breakdown:")
                for key, value in env_instance.episode_rewards.items():
                    if value != 0:
                        print(f"  {key}: {value:.3f}")
            
            if done[0]:
                break
        
        # Save results
        final_model_path = f"{save_path}/seven_rewards_model.zip"
        model.save(final_model_path)
        
        # Get final reward breakdown
        final_rewards = env_instance.episode_rewards.copy()
        
        results = {
            'reward_system': '7_components',
            'reward_breakdown': final_rewards,
            'timesteps': 5000,
            'training_time': training_time,
            'model_path': final_model_path,
            'status': 'success'
        }
        
        save_training_results(results, save_path)
        
        print("\n" + "=" * 80)
        print("✅ 7-REWARD SYSTEM TEST PASSED!")
        print("=" * 80)
        print("Final reward breakdown:")
        for key, value in final_rewards.items():
            if value != 0 or key == 'total':
                print(f"  {key}: {value:.3f}")
        print(f"\nModel: {final_model_path}")
        print("Ready for full training with complete reward system!")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return False

if __name__ == "__main__":
    test_seven_reward_system()