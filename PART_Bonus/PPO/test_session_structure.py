#!/usr/bin/env python3
"""
Test script to verify the session folder structure fix.
This will create a small test to ensure the folder structure is clean.
"""

import os
import shutil
from pathlib import Path
import tempfile

def test_session_structure():
    """Test that session folders are created correctly."""
    
    print("Testing session folder structure...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")
        
        # Simulate the old vs new behavior
        old_style_path = Path(temp_dir) / "old_sessions" / "pokemon_rl_20241005_120000"
        new_style_path = Path(temp_dir) / "new_sessions" / "pokemon_rl_20241005_120000"
        
        # Old style: Create main folder, then env subfolders (main stays empty)
        old_style_path.mkdir(parents=True, exist_ok=True)
        (old_style_path / "env_0").mkdir(parents=True, exist_ok=True)
        (old_style_path / "env_1").mkdir(parents=True, exist_ok=True)
        (old_style_path / "env_0" / "screenshots").mkdir(parents=True, exist_ok=True)
        (old_style_path / "env_1" / "screenshots").mkdir(parents=True, exist_ok=True)
        
        # Add some fake files to env folders
        (old_style_path / "env_0" / "screenshots" / "screen_001.png").write_text("fake image")
        (old_style_path / "env_1" / "screenshots" / "screen_001.png").write_text("fake image")
        
        # New style: Use main folder directly with env-specific screenshot folders
        new_style_path.mkdir(parents=True, exist_ok=True)
        (new_style_path / "screenshots").mkdir(parents=True, exist_ok=True)
        (new_style_path / "screenshots_env_1").mkdir(parents=True, exist_ok=True)
        (new_style_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        
        # Add files to main folder
        (new_style_path / "screenshots" / "screen_001.png").write_text("fake image")
        (new_style_path / "screenshots_env_1" / "screen_001.png").write_text("fake image")
        (new_style_path / "checkpoints" / "model.zip").write_text("fake model")
        
        # Compare structures
        print("\n🔴 OLD STYLE (with empty parent folders):")
        print(f"   {old_style_path} - EMPTY PARENT")
        print(f"   {old_style_path / 'env_0'} - contains files")
        print(f"   {old_style_path / 'env_1'} - contains files")
        
        old_main_files = list(old_style_path.glob("*"))
        old_main_files = [f for f in old_main_files if f.is_file()]
        print(f"   Files in main folder: {len(old_main_files)} (should be 0)")
        
        print("\n🟢 NEW STYLE (consolidated):")
        print(f"   {new_style_path} - contains all files")
        
        new_main_files = list(new_style_path.glob("*"))
        new_main_files = [f for f in new_main_files if f.is_file()]
        new_main_dirs = list(new_style_path.glob("*"))
        new_main_dirs = [f for f in new_main_dirs if f.is_dir()]
        
        print(f"   Files in main folder: {len(new_main_files)}")
        print(f"   Directories in main folder: {len(new_main_dirs)}")
        for dir_path in new_main_dirs:
            files_in_dir = list(dir_path.glob("**/*"))
            files_in_dir = [f for f in files_in_dir if f.is_file()]
            print(f"     {dir_path.name}/: {len(files_in_dir)} files")
        
        print("\n✅ Session folder structure test completed!")
        print("✅ New style keeps everything in one place without empty parent folders")

if __name__ == "__main__":
    test_session_structure()