#!/usr/bin/env python3
"""
Test PyBoy window creation directly
"""

import os
import sys
from pathlib import Path

def test_pyboy_window():
    """Test PyBoy window creation directly."""
    
    print("🎮 Testing PyBoy Window Creation")
    print("=" * 40)
    
    try:
        # Set up environment variables
        os.environ['SDL_VIDEODRIVER'] = 'windows'  # Force Windows SDL driver
        
        from pyboy import PyBoy
        
        # Find ROM
        rom_path = 'PokemonRed.gb'
        if not os.path.exists(rom_path):
            print(f"❌ ROM not found: {rom_path}")
            return
        
        print(f"📁 ROM found: {rom_path}")
        print(f"🚀 Creating PyBoy with SDL2 window...")
        
        # Create PyBoy with explicit window
        pyboy = PyBoy(
            rom_path,
            window='SDL2',  # Force SDL2 window
            debug=False
        )
        
        print("✅ PyBoy created successfully!")
        print("🖼️  Window should be visible now!")
        print("   Running for 5 seconds then closing...")
        
        # Run for a few seconds
        for i in range(300):  # ~5 seconds at 60 FPS
            pyboy.tick()
            if i % 60 == 0:
                print(f"   Frame {i}/300")
        
        print("🔄 Closing PyBoy...")
        pyboy.stop()
        print("✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pyboy_window()