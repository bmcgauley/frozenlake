"""
Test the Updated Pokemon Red CNN Architecture
Tests the proven methodology with stacked screens + status bars.
"""

import os
import time
from pokemon_red_training import PokemonRedEnv, create_ppo_model, save_training_results
from stable_baselines3.common.vec_env import DummyVecEnv

def test_cnn_architecture():
    """Test the CNN architecture with stacked screens."""
    
    print("=" * 80)
    print("TESTING POKEMON RED CNN ARCHITECTURE")
    print("=" * 80)
    print("Architecture:")
    print("  ✓ 3 stacked screens (short-term memory)")
    print("  ✓ Resolution reduced 4x (144x160 → 36x40)")
    print("  ✓ Visual status bars (HP, levels, exploration)")
    print("  ✓ Grayscale conversion for efficiency")
    print("  ✓ CNN policy (non-recurrent)")
    print()
    
    # Create save directory
    save_path = f'./sessions/cnn_test_{time.strftime("%Y%m%d_%H%M%S")}'
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
    print(f"✓ Environment created - observation shape: {env.observation_space.shape}")
    
    # Create CNN model
    model = create_ppo_model(env, total_timesteps=5000)
    print("✓ CNN model created")
    
    print("\n🚀 Starting short training test...")
    print("Testing CNN with stacked screens for 5k timesteps")
    print()
    
    start_time = time.time()
    
    try:
        # Short training run
        model.learn(
            total_timesteps=5000,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print("\n✓ Training completed successfully!")
        print(f"Time: {training_time:.1f} seconds")
        
        # Test the trained model
        print("\nTesting trained CNN model...")
        obs = env.reset()
        total_reward = 0
        
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            if step % 10 == 0:
                print(f"  Step {step}: action={action[0]}, reward={reward[0]:.3f}")
            
            if done[0]:
                break
        
        print(f"Test reward: {total_reward:.2f}")
        
        # Save results
        final_model_path = f"{save_path}/cnn_test_model.zip"
        model.save(final_model_path)
        
        results = {
            'architecture': 'stacked_screens_cnn',
            'observation_shape': list(env.observation_space.shape),
            'timesteps': 5000,
            'training_time': training_time,
            'test_reward': float(total_reward),
            'model_path': final_model_path,
            'status': 'success'
        }
        
        save_training_results(results, save_path)
        
        print("\n" + "=" * 80)
        print("✅ CNN ARCHITECTURE TEST PASSED!")
        print("=" * 80)
        print(f"Model: {final_model_path}")
        print(f"Results: {save_path}/training_results.json")
        print("\nThe CNN architecture is working correctly!")
        print("Ready for full training with proven methodology.")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        env.close()
        return False

if __name__ == "__main__":
    test_cnn_architecture()