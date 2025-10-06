#!/usr/bin/env python3
"""
Quick test script to verify PyBoy visual demo works on Windows
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append('.')

def test_visual_demo():
    """Test that PyBoy can show a visual window on Windows."""
    
    print("🎮 Testing PyBoy Visual Demo on Windows")
    print("=" * 50)
    
    try:
        from pokemon_red_training import demo_model
        
        # Find any available model
        sessions_dir = Path('./sessions')
        model_path = None
        
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    final_model = session_dir / 'final_model.zip'
                    if final_model.exists():
                        model_path = str(final_model)
                        print(f"Found model: {model_path}")
                        break
        
        if not model_path:
            print("❌ No trained models found!")
            print("   Run some training first to create models.")
            return
        
        print(f"🚀 Starting 30-second visual demo...")
        print(f"   PyBoy window should appear!")
        print(f"   Press Ctrl+C to stop early")
        
        # Run a short demo to test visuals
        demo_model(model_path, show_visual=True, max_minutes=0.5)  # 30 seconds
        
        print("✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visual_demo()