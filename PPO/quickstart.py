"""
Pokemon Red RL - Automated Quick Start Script

This script automates the entire setup and training process:
1. Checks system requirements
2. Installs dependencies (optional)
3. Verifies ROM
4. Configures training based on your hardware
5. Starts training
6. Opens TensorBoard automatically

Usage:
    python quickstart.py                    # Interactive setup
    python quickstart.py --auto             # Automatic with defaults
    python quickstart.py --install-deps     # Install dependencies first
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse

print("=" * 80)
print("POKEMON RED RL - QUICK START WIZARD")
print("=" * 80)
print("\nThis wizard will guide you through setup and start training.\n")

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Pokemon Red RL Quick Start')
parser.add_argument('--auto', action='store_true',
                   help='Skip prompts and use recommended defaults')
parser.add_argument('--install-deps', action='store_true',
                   help='Install Python dependencies automatically')
parser.add_argument('--no-tensorboard', action='store_true',
                   help='Do not automatically open TensorBoard')
parser.add_argument('--timesteps', type=int, default=10_000_000,
                   help='Total training timesteps (default: 10M)')
args = parser.parse_args()

# ============================================================================
# STEP 1: Check Python Version
# ============================================================================

print("Step 1: Checking Python version...")
python_version = sys.version_info

if python_version.major == 3 and python_version.minor >= 10:
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"❌ Python 3.10+ required (you have {python_version.major}.{python_version.minor})")
    sys.exit(1)

# ============================================================================
# STEP 2: Install Dependencies
# ============================================================================

print("\nStep 2: Checking dependencies...")

def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

required_packages = {
    'torch': 'torch',
    'gymnasium': 'gymnasium',
    'stable_baselines3': 'stable-baselines3',
    'pyboy': 'pyboy',
    'numpy': 'numpy',
    'PIL': 'Pillow',
    'skimage': 'scikit-image',
    'matplotlib': 'matplotlib'
}

missing = []
for module, pip_name in required_packages.items():
    if not check_package(module):
        missing.append(pip_name)

if missing:
    print(f"⚠️  Missing packages: {', '.join(missing)}")
    
    if args.install_deps or (not args.auto and 
       input("Install missing dependencies? (y/n): ").lower() == 'y'):
        print("\nInstalling dependencies...")
        print("This may take several minutes...\n")
        
        # Install from requirements.txt if it exists
        if os.path.exists('requirements.txt'):
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        else:
            # Install individually
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
        
        print("\n✅ Dependencies installed!")
    else:
        print("\n❌ Cannot proceed without dependencies.")
        print("Install manually: pip install -r requirements.txt")
        sys.exit(1)
else:
    print("✅ All dependencies installed")

# ============================================================================
# STEP 3: Verify ROM
# ============================================================================

print("\nStep 3: Checking ROM file...")

ROM_PATH = 'PokemonRed.gb'

if not os.path.exists(ROM_PATH):
    print(f"❌ ROM not found: {ROM_PATH}")
    print("\nPlease place your Pokemon Red ROM file as 'PokemonRed.gb'")
    print("in the current directory and run this script again.")
    sys.exit(1)

file_size = os.path.getsize(ROM_PATH)
if file_size != 1048576:
    print(f"⚠️  ROM size is {file_size} bytes (expected 1,048,576)")
    if not args.auto:
        if input("Continue anyway? (y/n): ").lower() != 'y':
            sys.exit(1)

print(f"✅ ROM found: {ROM_PATH}")

# ============================================================================
# STEP 4: Hardware Detection & Configuration
# ============================================================================

print("\nStep 4: Detecting hardware and configuring training...")

try:
    import psutil
    cpu_count = os.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"CPU Cores: {cpu_count}")
    print(f"RAM: {ram_gb:.1f} GB")
    
    # Determine optimal configuration
    if ram_gb >= 32 and cpu_count >= 16:
        recommended_envs = min(cpu_count, 24)
        config_tier = "HIGH-END"
    elif ram_gb >= 16 and cpu_count >= 8:
        recommended_envs = min(cpu_count, 16)
        config_tier = "STANDARD"
    else:
        recommended_envs = min(cpu_count // 2, 8)
        config_tier = "BUDGET"
    
    print(f"\nDetected configuration: {config_tier}")
    print(f"Recommended parallel environments: {recommended_envs}")
    
    if not args.auto:
        custom = input(f"\nUse {recommended_envs} environments? (y/n, or enter custom number): ")
        if custom.lower() == 'n':
            try:
                recommended_envs = int(input("Enter number of environments: "))
            except ValueError:
                print("Invalid number, using recommended value")
        elif custom.isdigit():
            recommended_envs = int(custom)
    
    num_envs = recommended_envs

except ImportError:
    print("⚠️  Cannot detect hardware (psutil not installed)")
    num_envs = 8  # Safe default

print(f"\n✅ Configuration: {num_envs} parallel environments")

# ============================================================================
# STEP 5: Create Training Configuration
# ============================================================================

print("\nStep 5: Creating training configuration...")

# Create a custom config file
config_content = f"""
# Pokemon Red RL Training Configuration
# Generated by quickstart.py

# Hardware-optimized settings
NUM_ENVS = {num_envs}
TOTAL_TIMESTEPS = {args.timesteps}
SAVE_FREQ = 50_000

# Training hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 1
GAMMA = 0.999
GAE_LAMBDA = 0.95

# Environment settings
ROM_PATH = '{ROM_PATH}'
HEADLESS = True
ACTION_FREQ = 24
MAX_STEPS = 8192
SAVE_SCREENSHOTS = True
"""

config_path = Path('training_config.py')
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"✅ Configuration saved: {config_path}")

# ============================================================================
# STEP 6: Create Directories
# ============================================================================

print("\nStep 6: Creating directories...")

directories = ['sessions', 'visualizations', 'eval_screenshots', 'shared_weights']
for dir_name in directories:
    Path(dir_name).mkdir(exist_ok=True)
    print(f"  ✅ {dir_name}/")

# ============================================================================
# STEP 7: Start Training
# ============================================================================

print("\n" + "=" * 80)
print("READY TO START TRAINING")
print("=" * 80)

print(f"\nConfiguration Summary:")
print(f"  Parallel Environments: {num_envs}")
print(f"  Total Timesteps: {args.timesteps:,}")
print(f"  Expected Duration: ~{args.timesteps / (num_envs * 50) / 3600:.1f} hours")
print(f"  Checkpoints: Every 50,000 steps")
print(f"  Screenshots: Automatic on milestones")

if not args.auto:
    print("\nPress Ctrl+C at any time to stop training (checkpoints will be saved)")
    input("\nPress ENTER to start training... ")

print("\n🚀 Starting training!\n")
print("=" * 80)

# Start training in subprocess
import subprocess
import threading

training_process = subprocess.Popen(
    [sys.executable, 'pokemon_red_training.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Stream output
def stream_output(process):
    for line in process.stdout:
        print(line, end='')

output_thread = threading.Thread(target=stream_output, args=(training_process,))
output_thread.daemon = True
output_thread.start()

# Wait a few seconds for training to initialize
time.sleep(5)

# ============================================================================
# STEP 8: Open TensorBoard
# ============================================================================

if not args.no_tensorboard:
    print("\n" + "=" * 80)
    print("Opening TensorBoard for monitoring...")
    print("=" * 80)
    
    # Find the latest session directory
    sessions_dir = Path('sessions')
    if sessions_dir.exists():
        session_dirs = sorted(sessions_dir.glob('pokemon_rl_*'))
        if session_dirs:
            latest_session = session_dirs[-1]
            tensorboard_dir = latest_session / 'tensorboard'
            
            print(f"\nTensorBoard: http://localhost:6006")
            print(f"Session: {latest_session.name}")
            
            try:
                # Start TensorBoard in background
                tensorboard_process = subprocess.Popen(
                    [sys.executable, '-m', 'tensorboard.main', 
                     '--logdir', str(tensorboard_dir),
                     '--host', 'localhost',
                     '--port', '6006'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                print("\n✅ TensorBoard started!")
                print("   Open your browser to: http://localhost:6006")
                
                # Try to open browser automatically
                try:
                    import webbrowser
                    time.sleep(2)
                    webbrowser.open('http://localhost:6006')
                except:
                    pass
                
            except Exception as e:
                print(f"⚠️  Could not start TensorBoard: {e}")
                print("   Start manually: tensorboard --logdir", tensorboard_dir)

# ============================================================================
# WAIT FOR TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING IN PROGRESS")
print("=" * 80)
print("\nMonitoring:")
print("  📊 TensorBoard: http://localhost:6006")
print("  📁 Checkpoints: sessions/pokemon_rl_TIMESTAMP/checkpoints/")
print("  📸 Screenshots: sessions/pokemon_rl_TIMESTAMP/screenshots/")
print("\nPress Ctrl+C to stop training")
print("=" * 80 + "\n")

try:
    training_process.wait()
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    training_process.terminate()
    training_process.wait()

print("\n" + "=" * 80)
print("TRAINING SESSION COMPLETE")
print("=" * 80)

# ============================================================================
# STEP 9: Generate Visualizations
# ============================================================================

print("\nGenerating training visualizations...")

try:
    # Find latest session
    sessions_dir = Path('sessions')
    session_dirs = sorted(sessions_dir.glob('pokemon_rl_*'))
    
    if session_dirs:
        latest_session = session_dirs[-1]
        print(f"Latest session: {latest_session.name}")
        
        # Run visualization script
        subprocess.run([
            sys.executable, 'visualize_training.py',
            '--session', str(latest_session),
            '--output', 'visualizations'
        ])
        
        print("\n✅ Visualizations generated in: visualizations/")
    
except Exception as e:
    print(f"⚠️  Could not generate visualizations: {e}")
    print("Generate manually: python visualize_training.py --session SESSIONPATH")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SETUP COMPLETE!")
print("=" * 80)

print("\n📋 Next Steps:")
print("\n1. View training progress:")
print("   - TensorBoard: http://localhost:6006")
print("   - Check visualizations/ folder for charts")

print("\n2. Test trained model:")
print("   python evaluate_model.py --model sessions/SESSION/checkpoints/best_model.zip --mode demo")

print("\n3. Resume training:")
print("   Edit pokemon_red_training.py to load checkpoint and run again")

print("\n4. Compare multiple runs:")
print("   python visualize_training.py --compare session1/ session2/ session3/")

print("\n📚 Documentation:")
print("   - README.md for detailed guide")
print("   - Check sessions/SESSION/screenshots/ for milestone images")
print("   - TensorBoard for real-time metrics")

print("\n🎮 Happy training!")
print("=" * 80)