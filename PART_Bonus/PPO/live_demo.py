#!/usr/bin/env python3
"""
Pokemon Red Live Demo Launcher
==============================

This script launches a live visual demo of your trained Pokemon Red AI.
The PyBoy game window will appear and you can watch the AI play the game.

Usage: python live_demo.py [model_path] [minutes]

Examples:
  python live_demo.py                           # Use latest model, no time limit
  python live_demo.py sessions/model.zip        # Use specific model
  python live_demo.py sessions/model.zip 5      # Run for 5 minutes
"""

import sys
from pathlib import Path

def find_latest_model():
    """Find the most recent trained model."""
    sessions_dir = Path('./sessions')
    
    if not sessions_dir.exists():
        return None
    
    # Look for final models first, then best models
    for model_type in ['final_model.zip', 'best_model.zip']:
        for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
            if session_dir.is_dir():
                model_path = session_dir / model_type
                if model_path.exists():
                    return str(model_path)
    
    return None

def main():
    """Launch live demo with visual PyBoy window."""
    
    print("🎮 Pokemon Red Live Demo Launcher")
    print("=" * 50)
    
    # Parse command line arguments
    model_path = None
    max_minutes = None
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
    if len(sys.argv) > 2:
        try:
            max_minutes = float(sys.argv[2])
        except ValueError:
            print(f"❌ Invalid time limit: {sys.argv[2]}")
            return
    
    # Find model if not specified
    if not model_path:
        print("🔍 Looking for latest trained model...")
        model_path = find_latest_model()
        
        if not model_path:
            print("❌ No trained models found!")
            print("   Train a model first using:")
            print("   python train_pokemon.py")
            return
        else:
            print(f"✅ Found model: {model_path}")
    
    # Verify model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("\n🚀 Launching Live Demo...")
    print("📋 What to expect:")
    print("   • PyBoy game window will open")
    print("   • You'll see Pokemon Red running")
    print("   • AI will control the game automatically")
    print("   • Press Ctrl+C to stop anytime")
    
    if max_minutes:
        print(f"   • Demo will run for {max_minutes} minutes")
    else:
        print("   • Demo will run until you stop it")
    
    print("\n🎯 Troubleshooting:")
    print("   • If no window appears, check your taskbar")
    print("   • Try Alt+Tab to find the PyBoy window")
    print("   • Window title should be 'PyBoy'")
    
    print("\n" + "=" * 50)
    
    try:
        from pokemon_red_training import demo_model
        
        # Launch the demo
        demo_model(model_path, show_visual=True, max_minutes=max_minutes)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the correct directory")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n🔧 Try these solutions:")
        print("   1. Make sure PokemonRed.gb ROM file exists")
        print("   2. Check that all dependencies are installed")
        print("   3. Verify the model file isn't corrupted")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()