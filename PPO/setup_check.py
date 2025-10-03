"""
Pokemon Red RL - Setup Verification Script

This script verifies that your system is properly configured for training:
- Checks all required dependencies
- Verifies ROM file
- Tests PyBoy emulator
- Validates hardware resources
- Creates necessary directories

Run this BEFORE starting training to catch issues early.

Usage:
    python setup_check.py
"""

import sys
import os
import hashlib
from pathlib import Path

print("=" * 80)
print("POKEMON RED RL - SETUP VERIFICATION")
print("=" * 80)
print("\nChecking your system configuration...\n")

# Track overall status
all_checks_passed = True
warnings = []

# ============================================================================
# CHECK 1: Python Version
# ============================================================================

print("1. Checking Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major == 3 and python_version.minor >= 10:
    print("   ✅ Python version OK")
elif python_version.major == 3 and python_version.minor >= 8:
    print("   ⚠️  Python 3.10+ recommended (you have 3.{})".format(python_version.minor))
    warnings.append("Consider upgrading to Python 3.10 or 3.11")
else:
    print("   ❌ Python 3.10+ required")
    all_checks_passed = False

# ============================================================================
# CHECK 2: Required Dependencies
# ============================================================================

print("\n2. Checking required dependencies...")

dependencies = {
    'torch': 'PyTorch',
    'gymnasium': 'Gymnasium',
    'stable_baselines3': 'Stable-Baselines3',
    'pyboy': 'PyBoy',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'skimage': 'scikit-image',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'pandas': 'Pandas'
}

missing_deps = []
installed_deps = []

for module_name, display_name in dependencies.items():
    try:
        if module_name == 'PIL':
            import PIL
            version = PIL.__version__
        elif module_name == 'skimage':
            import skimage
            version = skimage.__version__
        else:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
        
        print(f"   ✅ {display_name}: {version}")
        installed_deps.append(display_name)
    except ImportError:
        print(f"   ❌ {display_name}: NOT INSTALLED")
        missing_deps.append(display_name)
        all_checks_passed = False

if missing_deps:
    print(f"\n   Missing dependencies: {', '.join(missing_deps)}")
    print("   Install with: pip install " + " ".join([
        'torch', 'gymnasium', 'stable-baselines3', 'pyboy',
        'numpy', 'Pillow', 'scikit-image', 'matplotlib', 'seaborn', 'pandas'
    ]))

# ============================================================================
# CHECK 3: PyTorch Configuration
# ============================================================================

print("\n3. Checking PyTorch configuration...")

try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
        print("   ✅ GPU acceleration available")
    else:
        print("   ⚠️  No GPU detected - training will use CPU")
        print("   This will be significantly slower but will still work")
        warnings.append("Consider using a system with CUDA GPU for faster training")
    
except Exception as e:
    print(f"   ❌ Error checking PyTorch: {e}")
    all_checks_passed = False

# ============================================================================
# CHECK 4: ROM File
# ============================================================================

print("\n4. Checking ROM file...")

ROM_PATH = 'PokemonRed.gb'
EXPECTED_SIZE = 1048576  # 1MB
EXPECTED_SHA1 = 'ea9bcae617fdf159b045185467ae58b2e4a48b9a'

if os.path.exists(ROM_PATH):
    # Check file size
    file_size = os.path.getsize(ROM_PATH)
    print(f"   File found: {ROM_PATH}")
    print(f"   Size: {file_size:,} bytes")
    
    if file_size == EXPECTED_SIZE:
        print("   ✅ File size correct (1MB)")
        
        # Check SHA1 hash
        print("   Verifying ROM integrity...")
        sha1 = hashlib.sha1()
        with open(ROM_PATH, 'rb') as f:
            while True:
                data = f.read(65536)  # Read in 64kb chunks
                if not data:
                    break
                sha1.update(data)
        
        rom_hash = sha1.hexdigest()
        print(f"   SHA1: {rom_hash}")
        
        if rom_hash == EXPECTED_SHA1:
            print("   ✅ ROM verified - authentic Pokemon Red")
        else:
            print("   ⚠️  ROM hash doesn't match expected value")
            print("   This may be a different version or hacked ROM")
            warnings.append("ROM verification failed - ensure you have Pokemon Red (US)")
    else:
        print(f"   ❌ File size incorrect (expected {EXPECTED_SIZE:,} bytes)")
        all_checks_passed = False
else:
    print(f"   ❌ ROM file not found: {ROM_PATH}")
    print("   Place 'PokemonRed.gb' in the current directory")
    all_checks_passed = False

# ============================================================================
# CHECK 5: PyBoy Functionality
# ============================================================================

print("\n5. Testing PyBoy emulator...")

if os.path.exists(ROM_PATH):
    try:
        from pyboy import PyBoy
        
        # Try to initialize PyBoy
        print("   Initializing emulator...")
        pyboy = PyBoy(ROM_PATH, window_type='headless', disable_input=False)
        
        # Run a few ticks
        for _ in range(100):
            pyboy.tick()
        
        # Check if we can read memory
        test_byte = pyboy.memory[0xD35E]  # Map ID address
        
        pyboy.stop()
        
        print("   ✅ PyBoy emulator working correctly")
        print(f"   Successfully emulated 100 frames")
        
    except Exception as e:
        print(f"   ❌ PyBoy error: {e}")
        print("   Try reinstalling: pip install --upgrade pyboy")
        all_checks_passed = False
else:
    print("   ⏭️  Skipped (ROM not found)")

# ============================================================================
# CHECK 6: Hardware Resources
# ============================================================================

print("\n6. Checking hardware resources...")

try:
    import psutil
    
    # CPU info
    cpu_count = os.cpu_count()
    print(f"   CPU cores: {cpu_count}")
    
    if cpu_count >= 16:
        print("   ✅ Excellent CPU count for parallel training")
    elif cpu_count >= 8:
        print("   ✅ Good CPU count - training will work well")
    elif cpu_count >= 4:
        print("   ⚠️  Low CPU count - consider reducing NUM_ENVS")
        warnings.append("With <8 cores, reduce NUM_ENVS to 4-6 in training script")
    else:
        print("   ❌ Insufficient CPU cores - minimum 4 recommended")
        all_checks_passed = False
    
    # RAM info
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"   RAM: {ram_gb:.1f} GB total, {ram.available / (1024**3):.1f} GB available")
    
    if ram_gb >= 32:
        print("   ✅ Excellent RAM for training")
    elif ram_gb >= 16:
        print("   ✅ Sufficient RAM for training")
    elif ram_gb >= 8:
        print("   ⚠️  Limited RAM - reduce NUM_ENVS if you encounter issues")
        warnings.append("With <16GB RAM, reduce NUM_ENVS to 4-8")
    else:
        print("   ❌ Insufficient RAM - 16GB+ strongly recommended")
        all_checks_passed = False
    
    # Disk space
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / (1024**3)
    print(f"   Disk space: {disk_free_gb:.1f} GB free")
    
    if disk_free_gb >= 50:
        print("   ✅ Sufficient disk space")
    elif disk_free_gb >= 20:
        print("   ⚠️  Limited disk space - monitor usage during training")
        warnings.append("Free up disk space if training for extended periods")
    else:
        print("   ❌ Low disk space - 20GB+ recommended for training")
        all_checks_passed = False

except ImportError:
    print("   ⚠️  psutil not installed - cannot check hardware")
    print("   Install with: pip install psutil")

# ============================================================================
# CHECK 7: Directory Structure
# ============================================================================

print("\n7. Creating required directories...")

directories = [
    'sessions',
    'visualizations',
    'eval_screenshots'
]

for dir_name in directories:
    dir_path = Path(dir_name)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        print(f"   ✅ Created: {dir_name}/")
    else:
        print(f"   ✅ Exists: {dir_name}/")

# ============================================================================
# CHECK 8: File Permissions
# ============================================================================

print("\n8. Checking file permissions...")

test_files = [
    'pokemon_red_training.py',
    'evaluate_model.py',
    'visualize_training.py'
]

for file_name in test_files:
    if os.path.exists(file_name):
        if os.access(file_name, os.R_OK):
            print(f"   ✅ {file_name} - readable")
        else:
            print(f"   ❌ {file_name} - not readable")
            all_checks_passed = False
    else:
        print(f"   ⚠️  {file_name} - not found")
        warnings.append(f"Missing script: {file_name}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

if all_checks_passed and len(warnings) == 0:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nYour system is ready for training!")
    print("\nNext steps:")
    print("1. Run: python pokemon_red_training.py")
    print("2. Monitor with: tensorboard --logdir ./sessions/")
    print("3. Check progress in TensorBoard at http://localhost:6006")
    
elif all_checks_passed and len(warnings) > 0:
    print("\n✅ SETUP COMPLETE (with warnings)")
    print("\nWarnings:")
    for i, warning in enumerate(warnings, 1):
        print(f"{i}. {warning}")
    print("\nYou can proceed with training, but consider addressing warnings.")
    print("\nTo start training: python pokemon_red_training.py")
    
else:
    print("\n❌ SETUP INCOMPLETE")
    print("\nCritical issues found. Please fix the errors above before training.")
    print("\nCommon fixes:")
    print("1. Install missing dependencies:")
    print("   pip install torch gymnasium stable-baselines3 pyboy numpy Pillow")
    print("   pip install scikit-image matplotlib seaborn pandas")
    print("2. Place Pokemon Red ROM as 'PokemonRed.gb' in current directory")
    print("3. Ensure Python 3.10+ is installed")
    
    if len(warnings) > 0:
        print("\nAdditional warnings:")
        for i, warning in enumerate(warnings, 1):
            print(f"{i}. {warning}")

print("\n" + "=" * 80)

# ============================================================================
# RECOMMENDED CONFIGURATION
# ============================================================================

print("\n📋 RECOMMENDED CONFIGURATION FOR YOUR SYSTEM")
print("=" * 80)

try:
    cpu_count = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Calculate recommended settings
    if ram_gb >= 32 and cpu_count >= 16:
        rec_envs = min(cpu_count, 24)
        rec_steps = 2048
        rec_batch = 512
        tier = "HIGH-END"
    elif ram_gb >= 16 and cpu_count >= 8:
        rec_envs = min(cpu_count, 16)
        rec_steps = 2048
        rec_batch = 512
        tier = "STANDARD"
    else:
        rec_envs = min(cpu_count, 8)
        rec_steps = 1024
        rec_batch = 256
        tier = "BUDGET"
    
    print(f"\nSystem Tier: {tier}")
    print(f"\nRecommended settings:")
    print(f"  NUM_ENVS = {rec_envs}  # Parallel environments")
    print(f"  N_STEPS = {rec_steps}  # Steps per update")
    print(f"  BATCH_SIZE = {rec_batch}  # Batch size")
    print(f"  max_steps = 8192  # Episode length")
    
    print(f"\nExpected performance:")
    print(f"  Training speed: ~{rec_envs * 50} steps/sec")
    print(f"  RAM usage: ~{rec_envs * 1.5:.1f} GB")
    print(f"  Time to 10M steps: ~{10_000_000 / (rec_envs * 50) / 3600:.1f} hours")
    
    print("\nTo use these settings, edit pokemon_red_training.py:")
    print(f"  NUM_ENVS = {rec_envs}")
    print(f"  N_STEPS = {rec_steps}")
    print(f"  BATCH_SIZE = {rec_batch}")

except:
    print("\n⚠️  Cannot calculate recommendations - psutil not available")

print("\n" + "=" * 80)
print("Setup verification complete!")
print("=" * 80)