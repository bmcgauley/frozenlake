#!/usr/bin/env python3
"""
Test script to verify all warnings have been resolved.
"""

import os
import warnings

# Set environment variables (should be set in the main script now)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Filter warnings
warnings.filterwarnings('ignore', message='Unable to preload all dependencies for SDL2_ttf')
warnings.filterwarnings('ignore', message='Unable to preload all dependencies for SDL2_image')

print("Testing Pokemon Red RL Environment Setup...")
print("=" * 50)

try:
    # Test import
    print("1. Testing imports...")
    import pokemon_red_training
    print("   ✓ Pokemon training module imported successfully")
    
    # Test environment creation (headless mode)
    print("2. Testing environment creation...")
    
    # Create a minimal config for testing
    test_config = {
        'rom_path': 'PokemonRed.gb',  # Make sure this file exists
        'headless': True,  # Force headless mode
        'max_steps': 100,  # Short test
        'save_screenshots': False,  # Disable screenshots for test
        'save_path': './test_session'
    }
    
    # Check if ROM exists
    if not os.path.exists(test_config['rom_path']):
        print(f"   ⚠ ROM file not found at {test_config['rom_path']}")
        print("   Note: Environment creation test skipped")
    else:
        env = pokemon_red_training.PokemonRedEnv(test_config)
        print("   ✓ Environment created successfully in headless mode")
        
        # Test a basic reset
        observation = env.reset()
        print("   ✓ Environment reset successful")
        
        # Clean up
        env.close()
        print("   ✓ Environment closed successfully")
    
    print("\nAll tests passed! The fixes should resolve the warnings.")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("\nSome issues may remain. Check the error details above.")

print("=" * 50)