#!/usr/bin/env python3
"""
Test CNN Input Debugging System

This script creates a minimal test to verify the CNN debugging visualization works.
It creates a few test environments and captures some frames to ensure everything is working.
"""

import os
import sys
from pathlib import Path
import tempfile

# Add current directory to path for imports
sys.path.append('.')

def test_cnn_debugging():
    """Test the CNN debugging system with a minimal example."""
    
    print("🧪 Testing CNN Input Debugging System")
    print("=" * 50)
    
    # Check if ROM exists
    rom_path = 'PokemonRed.gb'
    if not os.path.exists(rom_path):
        print(f"❌ Pokemon Red ROM not found at: {rom_path}")
        print("This test requires the ROM file to create the environment")
        return False
    
    try:
        # Import the training module
        from pokemon_red_training import PokemonRedEnv
        
        print("✅ Successfully imported PokemonRedEnv")
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temporary directory: {temp_dir}")
            
            # Configure environment with CNN debugging enabled
            test_config = {
                'rom_path': rom_path,
                'headless': True,
                'action_freq': 24,
                'max_steps': 100,  # Short test
                'save_path': temp_dir,
                'save_screenshots': False,  # Disable regular screenshots for test
                'debug_cnn_input': True,    # 🔬 Enable CNN debugging
                'cnn_save_frequency': 5,    # Save every 5 steps for faster testing
                'env_rank': 0
            }
            
            print("🎮 Creating environment with CNN debugging enabled...")
            env = PokemonRedEnv(test_config)
            
            print("✅ Environment created successfully")
            print(f"📏 Observation space: {env.observation_space.shape}")
            print(f"🎯 Action space: {env.action_space.n}")
            
            # Test environment reset
            print("\n🔄 Testing environment reset...")
            observation, info = env.reset()
            print(f"✅ Reset successful, observation shape: {observation.shape}")
            
            # Take a few test steps
            print("\n👾 Taking test steps with CNN debugging...")
            total_steps = 20
            
            for step in range(total_steps):
                # Take random action
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                
                if step % 5 == 0:
                    print(f"   Step {step}: Action={action}, Reward={reward:.3f}, Shape={observation.shape}")
                
                if terminated or truncated:
                    print(f"   Episode ended at step {step}")
                    break
            
            # Check if CNN debug files were created
            cnn_debug_dir = Path(temp_dir) / 'cnn_debug'
            if cnn_debug_dir.exists():
                debug_images = list(cnn_debug_dir.glob("cnn_input_*.png"))
                debug_data = list(cnn_debug_dir.glob("cnn_data_*.npy"))
                
                print(f"\n🔬 CNN Debug Results:")
                print(f"   📁 Debug directory: {cnn_debug_dir}")
                print(f"   🖼️  Images created: {len(debug_images)}")
                print(f"   📊 Data files created: {len(debug_data)}")
                
                if debug_images:
                    print(f"   ✅ First image: {debug_images[0].name}")
                    print(f"   ✅ Last image: {debug_images[-1].name}")
                
                if len(debug_images) >= 2:
                    print("   ✅ CNN debugging is working correctly!")
                    
                    # Test loading one of the debug data files
                    try:
                        import numpy as np
                        test_data = np.load(debug_data[0])
                        print(f"   ✅ Debug data shape: {test_data.shape}")
                        print(f"   ✅ Debug data range: [{test_data.min()}, {test_data.max()}]")
                    except Exception as e:
                        print(f"   ⚠️  Could not load debug data: {e}")
                else:
                    print("   ⚠️  Expected more debug images")
            else:
                print(f"\n❌ CNN debug directory not found: {cnn_debug_dir}")
                return False
            
            # Clean up
            env.pyboy.stop()
            print("\n🧹 Environment cleaned up")
            
        print("\n✅ CNN debugging test completed successfully!")
        print("\nTo use CNN debugging in training:")
        print("1. Set debug_cnn_input=True in ENV_CONFIG")
        print("2. Adjust cnn_save_frequency to control how often frames are saved")
        print("3. Use cnn_debug_analyzer.py to analyze the results")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cnn_debugging()
    sys.exit(0 if success else 1)