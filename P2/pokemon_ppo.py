import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyboy import PyBoy
import os
import time
import pickle
import multiprocessing as mp
from multiprocessing import Manager, Queue
import signal
from PIL import Image
from datetime import datetime
import psutil
from torch.utils.tensorboard import SummaryWriter

# ============================================================================
# SYSTEM RESOURCE DETECTION AND OPTIMIZATION
# ============================================================================

def detect_optimal_device():
    """Detect the best available device for training."""
    print("🔍 Detecting available compute devices...")

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {device_count} GPU(s) - {device_name}")
        return 'cuda'

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ MPS available: Apple Silicon GPU")
        return 'mps'

    # Check for Intel NPU (if available)
    try:
        # This is a basic check - in practice, NPU detection might need platform-specific code
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            print("✅ XPU available: Intel GPU/NPU")
            return 'xpu'
    except:
        pass

    # Check for DirectML (Windows ML) as fallback for some GPUs
    try:
        import torch_directml
        device = torch_directml.device()
        print("✅ DirectML available: Windows ML acceleration")
        return device
    except ImportError:
        pass

    # Default to CPU
    print("⚠️  No GPU acceleration available, using CPU")
    return 'cpu'

def detect_optimal_agent_count(device, visible_agents=1):
    """Determine optimal number of agents based on system resources."""
    print("🔍 Analyzing system resources for optimal agent count...")

    # Get CPU information
    cpu_count = mp.cpu_count()
    print(f"   CPU cores: {cpu_count}")

    # Get memory information
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(".1f")

    # Base agent count on CPU cores, but adjust for memory and device
    if device == 'cuda':
        # GPU training - can handle more agents since GPU does the heavy lifting
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(".1f")

        # Limit by GPU memory (each agent needs ~500MB GPU memory for PyBoy + model)
        max_by_gpu_memory = int(gpu_memory_gb / 0.5)

        # Also consider CPU cores for environment management
        base_agents = min(cpu_count * 2, max_by_gpu_memory)

    elif device == 'mps':
        # Apple Silicon - similar to GPU but more conservative
        base_agents = min(cpu_count * 1.5, int(memory_gb / 2))

    elif device == 'xpu':
        # Intel XPU - similar to CUDA
        base_agents = min(cpu_count * 2, int(memory_gb / 1.5))

    else:
        # CPU-only training - more conservative
        base_agents = min(cpu_count, int(memory_gb / 3))

    # Ensure minimum of 1 agent, maximum reasonable limit
    optimal_agents = max(1, min(base_agents, 5))  # Cap at 8 to prevent resource exhaustion

    # Reserve some cores for system and visible agents
    if visible_agents > 0:
        optimal_agents = max(1, optimal_agents - 1)  # Reserve 1 core for visible agent management

    print(f"   Optimal agent count: {optimal_agents} (device: {device})")
    return optimal_agents

def get_system_info():
    """Get comprehensive system information for logging."""
    info = {
        'cpu_count': mp.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'torch_version': torch.__version__,
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }

    if torch.cuda.is_available():
        info['cuda_devices'] = torch.cuda.device_count()
        info['cuda_device_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    return info

# ============================================================================
# AUTO-CONFIGURED HYPERPARAMETERS
# ============================================================================

# Detect optimal device and agent count
DEVICE = detect_optimal_device() 
NUM_AGENTS = 3 # detect_optimal_agent_count(DEVICE)
NUM_VISIBLE_AGENTS = 1  # Keep one visible agent for monitoring

# Adjust hyperparameters based on device
if DEVICE == 'cuda':
    # GPU-optimized settings with STRONG exploration bias
    BATCH_SIZE = 512  # Smaller batch = more updates = more exploration variety
    LEARNING_RATE = 0.0003  # Lower LR = slower convergence = longer exploration phase
    PPO_EPOCHS = 4  # Fewer epochs = less overfitting to current policy
elif DEVICE in ['mps', 'xpu']:
    # Accelerated device settings
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0005
    PPO_EPOCHS = 3
else:
    # CPU-only settings (conservative)
    BATCH_SIZE = 128  # Smaller batch for CPU memory
    LEARNING_RATE = 0.0005  # Lower learning rate for stability
    PPO_EPOCHS = 3  # Fewer epochs to speed up training

# ============================================================================
# HYPERPARAMETERS - PPO STYLE (EXPLORATION FOCUSED)
# ============================================================================

EPISODES_PER_AGENT = 5000
MAX_STEPS_PER_EPISODE = 15000  # Even longer episodes for more exploration time

# PPO hyperparameters - TUNED FOR MAXIMUM EXPLORATION
DISCOUNT_FACTOR = 0.97  # Higher = care more about future rewards = explore further
GAE_LAMBDA = 0.95  # Lower = more focus on immediate advantage = less smoothing
PPO_CLIP = 0.3  # Higher = allow bigger policy changes = more exploration
VALUE_COEF = 0.5  # Balanced value loss
ENTROPY_COEF = 0.001  # MUCH HIGHER = strong bonus for random actions = forced exploration

# Knowledge pooling settings - REDUCED to prevent premature convergence
POOLING_FREQUENCY = 25  # Pool LESS often to let agents explore independently
POOLING_ALPHA = 0.1     # Mix LESS knowledge (10% shared, 90% individual)

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_configuration_summary():
    """Print a summary of the auto-detected configuration."""
    system_info = get_system_info()

    print("\n" + "="*80)
    print("🤖 AUTO-CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Agents: {NUM_AGENTS} total, {NUM_VISIBLE_AGENTS} visible")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"PPO Epochs: {PPO_EPOCHS}")
    print()
    print("System Resources:")
    print(f"  CPU Cores: {system_info['cpu_count']}")
    print(".1f")
    print(f"  PyTorch: {system_info['torch_version']}")
    print(f"  Python: {system_info['python_version']}")

    if 'cuda_devices' in system_info:
        print(f"  CUDA GPUs: {system_info['cuda_devices']}")
        for i, name in enumerate(system_info['cuda_device_names']):
            print(f"    GPU {i}: {name}")

    print("="*80 + "\n")

# Print configuration on import
print_configuration_summary()

# Pokemon Red memory addresses
PLAYER_X_ADDRESS = 0xD362 
PLAYER_Y_ADDRESS = 0xD363
MAP_ID_ADDRESS = 0xD35E

# Additional Pokemon Red memory addresses for reward tracking
MENU_STATE_ADDRESS = 0xD0FB  # 0 = overworld, other = in menu
MONEY_ADDRESS = 0xD347     # Money (3 bytes, BCD format)
BADGES_ADDRESS = 0xD356    # Gym badges bitfield
ITEM_COUNT_ADDRESS = 0xD31D # Number of items in bag
ITEM_BAG_ADDRESS = 0xD31E  # Start of item bag data
POKEMON_COUNT_ADDRESS = 0xD163 # Number of pokemon in party
POKEMON_PARTY_ADDRESS = 0xD164 # Start of pokemon party data
ELITE_FOUR_FLAGS_ADDRESS = 0xD75A # Elite four completion flags
POKECENTER_FLAG_ADDRESS = 0xD72C  # Last pokecenter used flag
POKEMON_HP_ADDRESSES = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]  # HP for each party pokemon

# Battle and exploration addresses
BATTLE_STATE_ADDRESS = 0xD057  # 0 = not in battle, other = in battle
ENEMY_HP_ADDRESS = 0xCFE6     # Enemy pokemon current HP (2 bytes)
ENEMY_MAX_HP_ADDRESS = 0xCFE8 # Enemy pokemon max HP (2 bytes)
CURRENT_POKEMON_HP_ADDRESS = 0xD015  # Current pokemon HP in battle (2 bytes)
TILE_IN_FRONT_ADDRESS = 0xCFC6 # Tile ID the player is standing on
POKEDEX_OWNED_ADDRESS = 0xD2F7 # Start of pokedex "owned" data (19 bytes)
POKEDEX_SEEN_ADDRESS = 0xD30A  # Start of pokedex "seen" data (19 bytes)
WILD_BATTLE_FLAG_ADDRESS = 0xD12A # Flag indicating wild pokemon battle

# NEW: Dialog and NPC interaction addresses
TEXT_BOX_ACTIVE_ADDRESS = 0xC4CF       # 0 = no text, other = text showing
JOYPAD_DISABLE_ADDRESS = 0xC2FA        # Player control disabled (dialog/cutscene)
FACING_DIRECTION_ADDRESS = 0xC109       # 0=down, 4=up, 8=left, 12=right
SPRITE_STATE_START_ADDRESS = 0xC100    # NPC/sprite data (check tiles in front)
CURRENT_MAP_SCRIPT_ADDRESS = 0xCC4D    # Map script pointer (can detect events)

# Window positions for visible agents
WINDOW_POSITIONS = [
    (50, 50),
    (50, 50),
    (50, 50),
]

# ============================================================================
# SAVE STATE CREATION
# ============================================================================

def get_screen_hash(pyboy):
    """Get hash of screen."""
    screen = pyboy.screen.ndarray
    return hash(screen.tobytes())

def is_on_title_screen(pyboy):
    """Detect title screen."""
    screen = pyboy.screen.ndarray
    upper_third = screen[:60, :, :]
    white_pixels = np.sum(upper_third > 200)
    return white_pixels > 2000

def create_playable_save(rom_path, save_path='playable_state.state'):
    """Create save by playing through intro."""
    print("\n" + "="*80)
    print("CREATING SAVE STATE")
    print("="*80)
    
    pyboy = PyBoy(rom_path, window="SDL2")
    pyboy.set_emulation_speed(0)
    
    print("\n1. Booting ROM...")
    for _ in range(200):
        pyboy.tick()
    
    print("\n2. Getting past title screen...")
    attempts = 0
    while is_on_title_screen(pyboy) and attempts < 100:
        if attempts % 10 == 0:
            print(f"   Attempt {attempts}/100")
        pyboy.button_press('start')
        for _ in range(20):
            pyboy.tick()
        pyboy.button_release('start')
        for _ in range(10):
            pyboy.tick()
        attempts += 1
    
    if is_on_title_screen(pyboy):
        for _ in range(50):
            pyboy.button_press('a')
            for _ in range(20):
                pyboy.tick()
            pyboy.button_release('a')
            for _ in range(10):
                pyboy.tick()
    
    print("\n3. Skipping menus...")
    for i in range(200):
        if i % 50 == 0:
            print(f"   {i}/200")
        pyboy.button_press('a')
        for _ in range(15):
            pyboy.tick()
        pyboy.button_release('a')
        for _ in range(5):
            pyboy.tick()
    
    print("\n4. Testing game state...")
    initial_hash = get_screen_hash(pyboy)
    
    for _ in range(5):
        pyboy.button_press('down')
        for _ in range(10):
            pyboy.tick()
        pyboy.button_release('down')
        for _ in range(5):
            pyboy.tick()
    
    after_hash = get_screen_hash(pyboy)
    screen_changed = initial_hash != after_hash
    
    print(f"   Screen changed: {screen_changed}")
    
    print(f"\n5. Saving to {save_path}...")
    with open(save_path, 'wb') as f:
        pyboy.save_state(f)
    
    print("✓ Save state created!")
    print("="*80 + "\n")
    
    pyboy.set_emulation_speed(1)
    time.sleep(3)
    
    pyboy.stop()
    return screen_changed

# ============================================================================
# PPO ACTOR-CRITIC NETWORK
# ============================================================================

class ActorCritic(nn.Module):
    """PPO Actor-Critic network with enhanced state features."""
    def __init__(self, action_size=7, device='cpu', feature_size=12):
        super(ActorCritic, self).__init__()
        
        self.device = device
        self.feature_size = feature_size
        
        # Screen encoder
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Feature encoder for game state
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        self.conv_out_size = None
        self.body = None
        self.actor = None
        self.critic = None
        self.action_size = action_size
    
    def _get_conv_output(self, shape, device):
        """Calculate conv output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape, device=device)
            output = self.conv(dummy)
            return int(np.prod(output.size()))
    
    def forward(self, screen, features):
        # Ensure inputs are on the correct device
        screen = screen.to(self.device)
        features = features.to(self.device)
        
        # Initialize on first pass
        if self.conv_out_size is None:
            self.conv_out_size = self._get_conv_output(screen.shape[1:], self.device)
            
            # Combine conv output + feature encoding
            combined_size = self.conv_out_size + 128
            
            self.body = nn.Sequential(
                nn.Linear(combined_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(self.device)
            
            self.actor = nn.Linear(512, self.action_size).to(self.device)
            self.critic = nn.Linear(512, 1).to(self.device)
        
        # Process screen
        conv_out = self.conv(screen)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Process features
        feature_out = self.feature_encoder(features)
        
        # Combine
        x = torch.cat([conv_out, feature_out], dim=1)
        x = self.body(x)
        
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value

# ============================================================================
# KNOWLEDGE POOLING SYSTEM
# ============================================================================

class SharedModelManager:
    """Manages shared knowledge between agents through file-based model parameter averaging."""
    
    def __init__(self, save_dir='shared_models'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.agent_models_dir = os.path.join(save_dir, 'agent_models')
        os.makedirs(self.agent_models_dir, exist_ok=True)
        
    def save_agent_model(self, agent_id, model_state_dict, episode):
        """Save an agent's model for knowledge sharing."""
        # Move all tensors to CPU before saving to ensure consistency
        cpu_state_dict = {}
        for key, tensor in model_state_dict.items():
            cpu_state_dict[key] = tensor.cpu()
        
        filename = f'agent_{agent_id}_ep_{episode}.pth'
        filepath = os.path.join(self.agent_models_dir, filename)
        torch.save(cpu_state_dict, filepath)
        print(f"📚 Agent {agent_id}: Saved model for knowledge pooling (Episode {episode})")
    
    def get_available_models(self, current_agent_id):
        """Get list of available models from other agents."""
        models = []
        for filename in os.listdir(self.agent_models_dir):
            if filename.endswith('.pth'):
                # Extract agent_id from filename
                parts = filename.split('_')
                if len(parts) >= 2:
                    agent_id = int(parts[1])
                    if agent_id != current_agent_id:  # Don't include self
                        models.append(os.path.join(self.agent_models_dir, filename))
        return models
    
    def pool_knowledge(self, current_agent_id, current_model_state):
        """Average current model with other available models."""
        available_models = self.get_available_models(current_agent_id)
        
        if len(available_models) == 0:
            print(f"🔄 Agent {current_agent_id}: No other models available for pooling")
            return current_model_state
        
        print(f"🧠 Agent {current_agent_id}: Pooling knowledge from {len(available_models)} other agents")
        
        # Start with current model
        pooled_state = {}
        for key in current_model_state.keys():
            pooled_state[key] = current_model_state[key].clone()
        
        # Add other models
        valid_models = 0
        for model_path in available_models:
            try:
                other_state = torch.load(model_path, map_location='cpu')  # Load to CPU first
                valid_models += 1
                
                # Move all tensors to the same device as current model
                for key in pooled_state.keys():
                    if key in other_state:
                        # Move to same device as pooled_state tensors
                        other_tensor = other_state[key].to(pooled_state[key].device)
                        pooled_state[key] += other_tensor
            except Exception as e:
                print(f"⚠️  Failed to load {model_path}: {e}")
                continue
        
        # Average all models (current + others)
        total_models = valid_models + 1
        if total_models > 1:
            for key in pooled_state.keys():
                pooled_state[key] /= total_models
        
        # Mix with original model using alpha
        mixed_state = {}
        for key in current_model_state.keys():
            mixed_state[key] = (1 - POOLING_ALPHA) * current_model_state[key] + POOLING_ALPHA * pooled_state[key]
        
        print(f"✅ Agent {current_agent_id}: Applied knowledge from {valid_models} agents (α={POOLING_ALPHA})")
        return mixed_state
    
    def cleanup_old_models(self, keep_latest=5):
        """Clean up old model files to prevent disk space issues."""
        files = []
        for filename in os.listdir(self.agent_models_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.agent_models_dir, filename)
                files.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time, newest first
        files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old files
        for filepath, _ in files[keep_latest:]:
            try:
                os.remove(filepath)
            except:
                pass

# ============================================================================
# TENSORBOARD LOGGING
# ============================================================================

class TensorBoardManager:
    """Manages TensorBoard logging for training visualization."""
    def __init__(self, log_dir='tensorboard_logs', run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.log_dir = os.path.join(log_dir, run_name)
        self.agent_writers = {}
        self.global_writer = SummaryWriter(os.path.join(self.log_dir, 'global'))
        
        print(f"📊 TensorBoard logging initialized: {self.log_dir}")
        print(f"   To view: tensorboard --logdir={log_dir}")
    
    def get_agent_writer(self, agent_id):
        """Get or create writer for specific agent."""
        if agent_id not in self.agent_writers:
            agent_dir = os.path.join(self.log_dir, f'agent_{agent_id}')
            self.agent_writers[agent_id] = SummaryWriter(agent_dir)
        return self.agent_writers[agent_id]
    
    def log_episode(self, agent_id, episode, reward, steps, unique_positions, 
                   policy_loss=None, value_loss=None, entropy=None):
        """Log episode metrics for an agent."""
        writer = self.get_agent_writer(agent_id)
        
        # Core metrics
        writer.add_scalar('Episode/Reward', reward, episode)
        writer.add_scalar('Episode/Steps', steps, episode)
        writer.add_scalar('Episode/UniquePositions', unique_positions, episode)
        writer.add_scalar('Episode/RewardPerStep', reward / max(steps, 1), episode)
        
        # Training metrics (if available)
        if policy_loss is not None:
            writer.add_scalar('Loss/Policy', policy_loss, episode)
        if value_loss is not None:
            writer.add_scalar('Loss/Value', value_loss, episode)
        if entropy is not None:
            writer.add_scalar('Loss/Entropy', entropy, episode)
    
    def log_global_metrics(self, episode, avg_reward, avg_steps, total_unique_positions,
                          best_agent_id=None, best_reward=None):
        """Log aggregate metrics across all agents."""
        self.global_writer.add_scalar('Global/AverageReward', avg_reward, episode)
        self.global_writer.add_scalar('Global/AverageSteps', avg_steps, episode)
        self.global_writer.add_scalar('Global/TotalUniquePositions', total_unique_positions, episode)
        
        if best_agent_id is not None and best_reward is not None:
            self.global_writer.add_scalar('Global/BestAgentReward', best_reward, episode)
            self.global_writer.add_scalar('Global/BestAgentID', best_agent_id, episode)
    
    def log_game_state(self, agent_id, episode, money=0, badges=0, pokemon_count=0, 
                      items=0, pokedex_owned=0, battles_won=0):
        """Log Pokemon-specific game state metrics."""
        writer = self.get_agent_writer(agent_id)
        
        writer.add_scalar('GameState/Money', money, episode)
        writer.add_scalar('GameState/Badges', badges, episode)
        writer.add_scalar('GameState/PokemonCount', pokemon_count, episode)
        writer.add_scalar('GameState/Items', items, episode)
        writer.add_scalar('GameState/PokedexOwned', pokedex_owned, episode)
        writer.add_scalar('GameState/BattlesWon', battles_won, episode)
    
    def log_exploration(self, agent_id, episode, maps_visited=0, total_states_seen=0,
                       screen_hash_unique=0):
        """Log exploration metrics."""
        writer = self.get_agent_writer(agent_id)
        
        writer.add_scalar('Exploration/MapsVisited', maps_visited, episode)
        writer.add_scalar('Exploration/TotalStatesSeen', total_states_seen, episode)
        writer.add_scalar('Exploration/ScreenHashUnique', screen_hash_unique, episode)
    
    def log_reward_breakdown(self, agent_id, episode, reward_components):
        """Log detailed reward component breakdown."""
        writer = self.get_agent_writer(agent_id)
        
        for component_name, value in reward_components.items():
            writer.add_scalar(f'Rewards/{component_name}', value, episode)
    
    def log_system_metrics(self, episode, cpu_percent=0, memory_percent=0, gpu_memory_mb=0):
        """Log system resource usage."""
        self.global_writer.add_scalar('System/CPU_Percent', cpu_percent, episode)
        self.global_writer.add_scalar('System/Memory_Percent', memory_percent, episode)
        if gpu_memory_mb > 0:
            self.global_writer.add_scalar('System/GPU_Memory_MB', gpu_memory_mb, episode)
    
    def log_knowledge_pooling(self, agent_id, episode, agents_pooled=0):
        """Log knowledge pooling events."""
        writer = self.get_agent_writer(agent_id)
        writer.add_scalar('KnowledgePooling/AgentsPooled', agents_pooled, episode)
    
    def log_screen(self, agent_id, episode, screen_array, tag='Screen'):
        """Log game screen visualization."""
        writer = self.get_agent_writer(agent_id)
        # Convert screen to CHW format (channel, height, width)
        if len(screen_array.shape) == 2:  # Grayscale
            screen_chw = screen_array[np.newaxis, :, :]
        else:  # RGB
            screen_chw = np.transpose(screen_array, (2, 0, 1))
        writer.add_image(tag, screen_chw, episode, dataformats='CHW')
    
    def close_all(self):
        """Close all TensorBoard writers."""
        for writer in self.agent_writers.values():
            writer.close()
        self.global_writer.close()
        print("📊 TensorBoard writers closed")

# ============================================================================
# SCREEN CAPTURE FOR VISUALIZATION
# ============================================================================

class ScreenCaptureManager:
    """Captures and saves game screens for visualization."""
    def __init__(self, save_dir='screen_captures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.capture_count = 0
    
    def capture_screen(self, pyboy, agent_id, episode, step, position):
        """Capture current screen with overlay."""
        # Get screen
        screen = pyboy.screen.ndarray.copy()
        
        # Create PIL image
        img = Image.fromarray(screen.astype('uint8'))
        
        # Save with metadata in filename
        x, y, map_id = position
        filename = f'agent{agent_id}_ep{episode:04d}_step{step:05d}_map{map_id}_x{x}_y{y}.png'
        save_path = os.path.join(self.save_dir, filename)
        
        img.save(save_path)
        self.capture_count += 1
        
        return save_path

# ============================================================================
# POKEMON ENVIRONMENT
# ============================================================================

class PokemonEnv:
    """Pokemon Red environment with PPO interface."""
    
    def __init__(self, rom_path, agent_id, visible=False, window_pos=(0,0), 
                 progress_queue=None, screen_capture_manager=None, device='cpu', tensorboard_manager=None):
        self.agent_id = agent_id
        self.progress_queue = progress_queue
        self.screen_capture_manager = screen_capture_manager  # Re-enabled for milestones
        self.device = device  # Store device for tensor operations
        self.tensorboard_manager = tensorboard_manager  # TensorBoard logging
        
        # Reward tracking for detailed breakdown
        self.reward_components = {
            'exploration': 0,
            'battle': 0,
            'progression': 0,
            'penalty': 0,
            'movement': 0
        }
        self.battles_won = 0  # Track battle victories
        
        # Create PyBoy instance
        if visible:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_pos[0]},{window_pos[1]}"
            self.pyboy = PyBoy(rom_path, window="SDL2")
            print(f"  Agent {agent_id}: Creating VISIBLE window at {window_pos}")
        else:
            self.pyboy = PyBoy(rom_path, window="null")
        
        self.pyboy.set_emulation_speed(0)
        
        # Load save
        if not os.path.exists('playable_state.state'):
            raise FileNotFoundError("playable_state.state not found!")
        
        with open('playable_state.state', 'rb') as f:
            self.pyboy.load_state(f)
        
        time.sleep(0.1)
        
        # Actions
        self.actions = ['up', 'down', 'left', 'right', 'a', 'b', 'wait']
        
        # Position tracking
        self.visited_positions = {}
        self.position_history = []
        
        # Direction consistency tracking (to reduce spinning)
        self.last_direction = None
        self.direction_consistency_count = 0
        
        # Memory system - track visited maps and state hashes
        self.visited_maps = set()  # Track which maps we've been to
        self.map_visit_times = {}  # Track when we last visited each map
        self.state_memory = {}  # Map state hash -> (visit_count, last_visit_step)
        self.screen_hash_history = []  # Recent screen hashes to detect loops
        
        # Game state tracking for rewards
        self.previous_money = 0
        self.previous_badges = 0
        self.previous_item_count = 0
        self.previous_pokemon_count = 0
        self.previous_elite_four_flags = 0
        self.previous_pokemon_hp = [0] * 6
        self.visited_states = set()  # Track all unique game states for exploration penalty
        
        # Battle and exploration tracking
        self.previous_enemy_hp = 0
        self.previous_current_pokemon_hp = 0
        self.previous_pokedex_owned = 0
        self.in_battle_last_step = False
        self.battle_victory_detected = False
        
        # Stats
        self.steps = 0
        self.episode_reward = 0
        self.total_episodes = 0
        self.last_progress_report = time.time()
        
        # Initialize
        self._update_position_tracking()
        
        # Initialize game state tracking
        self.previous_money = self._get_money()
        self.previous_badges = self._get_badges()
        self.previous_item_count = self._get_item_count()
        self.previous_pokemon_count = self._get_pokemon_count()
        self.previous_elite_four_flags = self._get_elite_four_flags()
        self.previous_pokemon_hp = self._get_pokemon_hp()
        
        # Initialize battle and exploration tracking
        self.previous_enemy_hp = self._get_enemy_hp()
        self.previous_current_pokemon_hp = self._get_current_pokemon_battle_hp()
        self.previous_pokedex_owned = self._get_pokedex_owned_count()
        self.in_battle_last_step = self._is_in_battle()
        self.battle_victory_detected = False
        
        if visible:
            print(f"  Agent {agent_id}: Initialized with VISIBLE window")
        else:
            print(f"  Agent {agent_id}: Initialized (headless)")
    
    def _read_memory(self, address, default=0):
        """Safely read memory."""
        try:
            return self.pyboy.memory[address]
        except:
            return default
    
    def _get_position(self):
        """Get current player position."""
        x = self._read_memory(PLAYER_X_ADDRESS, 0)
        y = self._read_memory(PLAYER_Y_ADDRESS, 0)
        map_id = self._read_memory(MAP_ID_ADDRESS, 0)
        return x, y, map_id
    
    def _is_menu_open(self):
        """Check if a menu is currently open."""
        menu_state = self._read_memory(MENU_STATE_ADDRESS, 0)
        return menu_state != 0
    
    def _get_money(self):
        """Get current money (BCD format)."""
        # Pokemon Red stores money in BCD format across 3 bytes
        money_bytes = [self._read_memory(MONEY_ADDRESS + i, 0) for i in range(3)]
        # Convert BCD to decimal
        money = 0
        for byte_val in money_bytes:
            money = money * 100 + ((byte_val >> 4) * 10) + (byte_val & 0x0F)
        return money
    
    def _get_badges(self):
        """Get gym badges bitfield."""
        return self._read_memory(BADGES_ADDRESS, 0)
    
    def _get_item_count(self):
        """Get number of items in bag."""
        return self._read_memory(ITEM_COUNT_ADDRESS, 0)
    
    def _get_pokemon_count(self):
        """Get number of pokemon in party."""
        return self._read_memory(POKEMON_COUNT_ADDRESS, 0)
    
    def _get_elite_four_flags(self):
        """Get elite four completion flags."""
        return self._read_memory(ELITE_FOUR_FLAGS_ADDRESS, 0)
    
    def _get_pokemon_hp(self):
        """Get HP of all party pokemon."""
        hp_values = []
        for addr in POKEMON_HP_ADDRESSES:
            hp = (self._read_memory(addr, 0) << 8) | self._read_memory(addr + 1, 0)
            hp_values.append(hp)
        return hp_values
    
    def _get_game_state_hash(self):
        """Get a hash representing the current game state for exploration tracking."""
        x, y, map_id = self._get_position()
        menu_open = self._is_menu_open()
        in_battle = self._is_in_battle()
        # Create a simplified state representation
        state = (x, y, map_id, menu_open, in_battle)
        return hash(state)
    
    def _get_screen_hash(self):
        """Get hash of the current screen for visual memory."""
        screen = self.pyboy.screen.ndarray
        # Downsample to 40x36 for efficiency (from 160x144)
        small_screen = screen[::4, ::4, :]
        return hash(small_screen.tobytes())
    
    def _is_in_recent_state(self, state_hash, lookback=20):
        """Check if we've seen this state very recently (loop detection)."""
        if len(self.screen_hash_history) < lookback:
            return False
        return state_hash in self.screen_hash_history[-lookback:]
    
    def _is_in_battle(self):
        """Check if currently in a battle."""
        battle_state = self._read_memory(BATTLE_STATE_ADDRESS, 0)
        return battle_state != 0
    
    def _is_wild_battle(self):
        """Check if in a wild pokemon battle."""
        return self._read_memory(WILD_BATTLE_FLAG_ADDRESS, 0) != 0
    
    def _get_enemy_hp(self):
        """Get enemy pokemon current HP."""
        return (self._read_memory(ENEMY_HP_ADDRESS, 0) << 8) | self._read_memory(ENEMY_HP_ADDRESS + 1, 0)
    
    def _get_enemy_max_hp(self):
        """Get enemy pokemon max HP."""
        return (self._read_memory(ENEMY_MAX_HP_ADDRESS, 0) << 8) | self._read_memory(ENEMY_MAX_HP_ADDRESS + 1, 0)
    
    def _get_current_pokemon_battle_hp(self):
        """Get current pokemon HP in battle."""
        return (self._read_memory(CURRENT_POKEMON_HP_ADDRESS, 0) << 8) | self._read_memory(CURRENT_POKEMON_HP_ADDRESS + 1, 0)
    
    def _is_in_tall_grass(self):
        """Check if player is standing in tall grass or wild area."""
        tile_id = self._read_memory(TILE_IN_FRONT_ADDRESS, 0)
        # Common tall grass tile IDs in Pokemon Red (this is an approximation)
        # Tall grass tiles are typically in the range 0x14-0x17, 0x52-0x55
        return tile_id in [0x14, 0x15, 0x16, 0x17, 0x52, 0x53, 0x54, 0x55]
    
    def _get_pokedex_owned_count(self):
        """Get number of pokemon owned in pokedex."""
        owned_bytes = [self._read_memory(POKEDEX_OWNED_ADDRESS + i, 0) for i in range(19)]
        count = 0
        for byte_val in owned_bytes:
            # Count set bits in each byte
            count += bin(byte_val).count('1')
        return count
    
    def _is_in_dialog(self):
        """Check if dialog/text box is active."""
        text_active = self._read_memory(TEXT_BOX_ACTIVE_ADDRESS, 0) != 0
        joypad_disabled = self._read_memory(JOYPAD_DISABLE_ADDRESS, 0) != 0
        return text_active or joypad_disabled
    
    def _get_facing_direction(self):
        """Get direction player is facing (0-3)."""
        direction = self._read_memory(FACING_DIRECTION_ADDRESS, 0)
        # Convert to 0-3: 0=down, 1=up, 2=left, 3=right
        return min(direction // 4, 3)
    
    def _is_facing_npc(self):
        """Check if an NPC/sprite is directly in front of the player."""
        tile_front = self._read_memory(TILE_IN_FRONT_ADDRESS, 0)
        # NPC tiles are typically in the range 0x01-0x0F (varies by map)
        # This is a simplified check
        return 0x01 <= tile_front <= 0x0F
    
    def _is_in_pokecenter(self):
        """Check if inside a Pokemon Center."""
        map_id = self._read_memory(MAP_ID_ADDRESS, 0)
        # Pokemon Center map IDs in Pokemon Red (approximate)
        pokecenter_maps = [0xC6, 0xC7, 0xC8, 0xC9, 0xCA]
        return map_id in pokecenter_maps
    
    def _is_in_pokemart(self):
        """Check if inside a Pokemart."""
        map_id = self._read_memory(MAP_ID_ADDRESS, 0)
        # Pokemart map IDs in Pokemon Red (approximate)
        pokemart_maps = [0xCB, 0xCC, 0xCD, 0xCE, 0xCF]
        return map_id in pokemart_maps
    
    def _detect_battle_victory(self):
        """Detect if a battle was just won."""
        in_battle = self._is_in_battle()
        enemy_hp = self._get_enemy_hp()
        
        # Battle victory: was in battle last step, enemy had HP, now enemy HP is 0 or not in battle
        if self.in_battle_last_step and not in_battle:
            return True
        elif in_battle and self.previous_enemy_hp > 0 and enemy_hp == 0:
            return True
        return False
    
    def _detect_pokemon_capture(self):
        """Detect if a pokemon was just captured (new pokedex entry)."""
        current_owned = self._get_pokedex_owned_count()
        return current_owned > self.previous_pokedex_owned
    
    def _update_position_tracking(self):
        """Update position tracking."""
        x, y, map_id = self._get_position()
        pos_key = (x, y, map_id)
        
        if pos_key not in self.visited_positions:
            self.visited_positions[pos_key] = 0
        self.visited_positions[pos_key] += 1
        
        self.position_history.append((x, y, map_id))
    
    def _send_progress_update(self):
        """Send progress update to main process."""
        if self.progress_queue is not None:
            # Only send every 5 seconds to avoid queue overflow
            if time.time() - self.last_progress_report > 5:
                try:
                    update = {
                        'agent_id': self.agent_id,
                        'episode': self.total_episodes,
                        'steps': self.steps,
                        'reward': self.episode_reward,
                        'unique_positions': len(self.visited_positions),
                        'position': self._get_position()
                    }
                    self.progress_queue.put_nowait(update)
                    self.last_progress_report = time.time()
                except:
                    pass  # Queue full, skip
    
    def _capture_milestone(self, milestone_type):
        """Capture screenshot for major milestones."""
        if self.screen_capture_manager:
            x, y, map_id = self._get_position()
            # Create special filename for milestones
            filename = f'MILESTONE_{milestone_type}_agent{self.agent_id}_ep{self.total_episodes:04d}_step{self.steps:05d}_map{map_id}_x{x}_y{y}.png'
            
            # Get screen
            screen = self.pyboy.screen.ndarray.copy()
            from PIL import Image
            img = Image.fromarray(screen.astype('uint8'))
            
            # Save with milestone prefix
            import os
            save_path = os.path.join(self.screen_capture_manager.save_dir, filename)
            img.save(save_path)
            
            print(f"📸 Agent {self.agent_id}: Captured {milestone_type} milestone at {save_path}")
            return save_path
        return None
    
    def get_screen_state(self, device='cpu'):
        """Get screen as tensor."""
        screen = self.pyboy.screen.ndarray
        gray = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
        gray = gray / 255.0
        return torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    def get_enhanced_state(self, device='cpu'):
        """Get enhanced state with game context features."""
        # Original screen
        screen = self.get_screen_state(device)
        
        # Additional state features (normalized to [0, 1])
        x, y, map_id = self._get_position()
        features = torch.FloatTensor([
            x / 255.0,                                    # Player X (normalized)
            y / 255.0,                                    # Player Y (normalized)
            map_id / 255.0,                               # Map ID (normalized)
            float(self._is_in_battle()),                  # Battle flag
            float(self._is_menu_open()),                  # Menu flag
            float(self._is_in_dialog()),                  # Dialog flag
            self._get_pokemon_count() / 6.0,              # Pokemon count (0-6)
            bin(self._get_badges()).count('1') / 8.0,     # Badge count (0-8)
            self._get_facing_direction() / 3.0,           # Facing direction
            float(self._is_facing_npc()),                 # Facing NPC
            float(self._is_in_pokecenter()),              # In Pokecenter
            float(self._is_in_pokemart()),                # In Pokemart
        ]).unsqueeze(0).to(device)
        
        return screen, features
    
    def step(self, action_idx):
        """Execute action."""
        prev_x, prev_y, prev_map = self._get_position()
        prev_pos_key = (prev_x, prev_y, prev_map)
        
        # Execute action
        action_name = self.actions[action_idx]
        
        if action_name == 'wait':
            for _ in range(12):
                self.pyboy.tick()
        else:
            self.pyboy.button_press(action_name)
            for _ in range(10):
                self.pyboy.tick()
            self.pyboy.button_release(action_name)
            for _ in range(5):
                self.pyboy.tick()
        
        # Get new position
        new_x, new_y, new_map = self._get_position()
        new_pos_key = (new_x, new_y, new_map)
        
        # Update tracking
        self._update_position_tracking()
        
        # Calculate reward
        reward = self._calculate_reward(prev_pos_key, new_pos_key)
        
        # Update stats
        self.steps += 1
        self.episode_reward += reward
        
        # Send progress update
        self._send_progress_update()
        
        # Capture screen occasionally - DISABLED
        # if self.screen_capture_manager and self.steps % 500 == 0:
        #     self.screen_capture_manager.capture_screen(
        #         self.pyboy, self.agent_id, self.total_episodes, self.steps, new_pos_key
        #     )
        
        # Check done
        done = self.steps >= MAX_STEPS_PER_EPISODE
        
        next_state_screen, next_state_features = self.get_enhanced_state(self.device)
        
        info = {
            'position': new_pos_key,
            'unique_positions': len(self.visited_positions),
            'episode_reward': self.episode_reward
        }
        
        return (next_state_screen, next_state_features), reward, done, info
    
    def _calculate_reward(self, prev_pos_key, new_pos_key):
        """
        Generalized reward function based on universal game principles.
        Works for any Pokemon game or similar RPG, not hardcoded to specific maps.
        """
        reward = 0
        
        # Reset reward component tracking
        self.reward_components = {
            'exploration': 0,
            'battle': 0,
            'progression': 0,
            'penalty': 0,
            'movement': 0
        }
        
        # ===================================================================
        # HYPERPARAMETERS - EXPLORATION FOCUSED (Less penalties, more rewards)
        # ===================================================================
        LOOP_PENALTY = 0.5              # Reduced penalty for loops
        REVISIT_PENALTY_RATE = 0.01      # Much gentler revisit penalty
        MAX_REVISIT_PENALTY = 1.0        # Lower cap
        NEW_STATE_BONUS = 20.0           # HUGE reward for new discoveries
        DIALOG_PENALTY = 0.1             # Minimal dialog penalty
        MENU_PENALTY = 0.1               # Minimal menu penalty
        STUCK_PENALTY = 0.2              # Lower stuck penalty
        DIRECTION_CONSISTENCY_BONUS = 0.05  # Small bonus for consistency
        MAP_DISCOVERY_BONUS = 100.0      # MASSIVE reward for new maps
        MAP_REVISIT_QUICK_PENALTY = 5.0  # Decent penalty for map flipping
        MAP_REVISIT_LATE_BONUS = 2.0     # Good reward for returning later
        MAP_REVISIT_THRESHOLD = 2000     # Can return sooner
        BATTLE_REWARD = 0.75             # Higher battle reward
        BATTLE_DAMAGE_MULTIPLIER = 2.0   # Higher damage reward
        BATTLE_VICTORY_WILD = 200.0      # Higher victory reward
        BATTLE_VICTORY_TRAINER = 350.0   # Much higher trainer reward
        POKEMON_CAPTURE_BASE = 400.0     # Higher capture reward
        POKEMON_CAPTURE_MULTIPLIER = 2.0 # Higher multiplier
        BADGE_REWARD = 500.0             # HUGE badge reward
        ITEM_GAIN_REWARD = 10.0          # Higher item reward
        ITEM_USE_PENALTY = 0.05          # Minimal use penalty
        MONEY_GAIN_MULTIPLIER = 0.2      # Higher money reward
        MONEY_SPEND_PENALTY = 0.01       # Minimal spend penalty
        HEALING_BASE_REWARD = 1.0        # Higher healing reward
        HEALING_HP_MULTIPLIER = 0.05     # Higher HP reward
        TALL_GRASS_BONUS = 0.05           # Much higher grass bonus
        MOVEMENT_BONUS = 0.01             # Higher movement reward
        TIME_PENALTY = 0.01             # MUCH lower time penalty
        
        # ===================================================================
        # STATE TRACKING
        # ===================================================================
        
        # Get current game state
        current_money = self._get_money()
        current_badges = self._get_badges()
        current_item_count = self._get_item_count()
        current_pokemon_count = self._get_pokemon_count()
        current_elite_four_flags = self._get_elite_four_flags()
        current_pokemon_hp = self._get_pokemon_hp()
        screen_hash = self._get_screen_hash()
        menu_open = self._is_menu_open()
        in_dialog = self._is_in_dialog()
        
        # Update screen hash history (keep last 50 for loop detection)
        self.screen_hash_history.append(screen_hash)
        if len(self.screen_hash_history) > 50:
            self.screen_hash_history.pop(0)
        
        # Update state memory
        if screen_hash not in self.state_memory:
            self.state_memory[screen_hash] = (0, self.steps)
        visit_count, last_visit = self.state_memory[screen_hash]
        self.state_memory[screen_hash] = (visit_count + 1, self.steps)
        
        # ===================================================================
        # MEMORY-BASED REWARDS (Prevents exploits, encourages true exploration)
        # ===================================================================
        
        # Penalty for revisiting same screen state too quickly (loop detection)
        if self._is_in_recent_state(screen_hash, lookback=10):
            penalty = LOOP_PENALTY
            reward -= penalty
            self.reward_components['penalty'] += penalty
        
        # Penalty for revisiting any state we've seen before
        if visit_count > 0:
            revisit_penalty = min(REVISIT_PENALTY_RATE * visit_count, MAX_REVISIT_PENALTY)
            reward -= revisit_penalty
            self.reward_components['penalty'] += revisit_penalty
        else:
            # Reward for truly new state (first time seeing this screen)
            reward += NEW_STATE_BONUS
            self.reward_components['exploration'] += NEW_STATE_BONUS
        
        # ===================================================================
        # BEHAVIORAL PENALTIES (Discourage bad habits)
        # ===================================================================
        
        # Penalty for dialog/NPC interaction spam
        if in_dialog:
            reward -= DIALOG_PENALTY
            self.reward_components['penalty'] += DIALOG_PENALTY
        
        # Penalty for menu spam
        if menu_open:
            reward -= MENU_PENALTY
            self.reward_components['penalty'] += MENU_PENALTY
        
        # Penalty for not moving (spinning/stuck)
        if new_pos_key == prev_pos_key:
            reward -= STUCK_PENALTY
            self.reward_components['penalty'] += STUCK_PENALTY
        
        # ===================================================================
        # MOVEMENT QUALITY (Encourage purposeful navigation)
        # ===================================================================
        
        # Track movement direction for consistency
        if new_pos_key != prev_pos_key:
            dx = new_pos_key[0] - prev_pos_key[0]
            dy = new_pos_key[1] - prev_pos_key[1]
            current_direction = (dx, dy)
            
            if self.last_direction is not None and current_direction == self.last_direction:
                self.direction_consistency_count += 1
                reward += DIRECTION_CONSISTENCY_BONUS
            else:
                self.direction_consistency_count = 0
            
            self.last_direction = current_direction

        # ===================================================================
        # BATTLE SYSTEM (Universal combat rewards)
        # ===================================================================
        
        in_battle = self._is_in_battle()
        if in_battle:
            battle_reward = BATTLE_REWARD
            reward += battle_reward
            self.reward_components['battle'] += battle_reward
            
            # Reward for dealing damage (skill demonstration)
            enemy_hp = self._get_enemy_hp()
            if self.previous_enemy_hp > 0 and enemy_hp < self.previous_enemy_hp:
                damage_dealt = self.previous_enemy_hp - enemy_hp
                damage_reward = damage_dealt * BATTLE_DAMAGE_MULTIPLIER
                reward += damage_reward
                self.reward_components['battle'] += damage_reward
        
        # Battle victory detection (works for any game)
        if self._detect_battle_victory():
            if self._is_wild_battle():
                victory_reward = BATTLE_VICTORY_WILD
                reward += victory_reward
                self.reward_components['battle'] += victory_reward
                self._capture_milestone("BATTLE_WIN_WILD")
            else:
                victory_reward = BATTLE_VICTORY_TRAINER
                reward += victory_reward
                self.reward_components['battle'] += victory_reward
                self._capture_milestone("BATTLE_WIN_TRAINER")
            self.battle_victory_detected = True
            self.battles_won += 1
        
        # Pokemon capture (collecting is universal RPG mechanic)
        if self._detect_pokemon_capture():
            new_entries = self._get_pokedex_owned_count() - self.previous_pokedex_owned
            capture_reward = POKEMON_CAPTURE_BASE * new_entries * POKEMON_CAPTURE_MULTIPLIER
            reward += capture_reward
            self.reward_components['progression'] += capture_reward
            self._capture_milestone("POKEMON_CAPTURE")
        
        # Wild area exploration (encourages finding encounters)
        if self._is_in_tall_grass():
            grass_bonus = TALL_GRASS_BONUS
            reward += grass_bonus
            self.reward_components['exploration'] += grass_bonus
        
        # Update battle tracking
        self.in_battle_last_step = in_battle
        self.previous_enemy_hp = self._get_enemy_hp()
        self.previous_current_pokemon_hp = self._get_current_pokemon_battle_hp()
        self.previous_pokedex_owned = self._get_pokedex_owned_count()
        
        # ===================================================================
        # PROGRESSION MILESTONES (Universal goal achievements)
        # ===================================================================
        
        # Badge/achievement system (works for any game with achievements)
        badge_diff = bin(current_badges).count('1') - bin(self.previous_badges).count('1')
        if badge_diff > 0:
            badge_reward = BADGE_REWARD * badge_diff
            reward += badge_reward
            self.reward_components['progression'] += badge_reward
            self._capture_milestone(f"BADGE_{bin(current_badges).count('1')}")
        
        # End-game progression (elite four or equivalent)
        elite_four_diff = bin(current_elite_four_flags).count('1') - bin(self.previous_elite_four_flags).count('1')
        if elite_four_diff > 0:
            endgame_reward = BADGE_REWARD * 5 * elite_four_diff
            reward += endgame_reward
            self.reward_components['progression'] += endgame_reward
            self._capture_milestone(f"ENDGAME_{bin(current_elite_four_flags).count('1')}")
        
        # ===================================================================
        # RESOURCE MANAGEMENT (Universal RPG mechanics)
        # ===================================================================
        
        # Party size changes (collecting team members)
        pokemon_diff = current_pokemon_count - self.previous_pokemon_count
        if pokemon_diff > 0:
            reward += POKEMON_CAPTURE_BASE / 15 * pokemon_diff  # Smaller reward than capture
        elif pokemon_diff < 0:
            reward -= ITEM_GAIN_REWARD / 2 * abs(pokemon_diff)
        
        # Inventory changes (item collection)
        item_diff = current_item_count - self.previous_item_count
        if item_diff > 0:
            reward += ITEM_GAIN_REWARD * item_diff
        elif item_diff < 0:
            reward -= ITEM_USE_PENALTY * abs(item_diff)
        
        # Currency changes (economic progress)
        money_diff = current_money - self.previous_money
        if money_diff > 0:
            reward += MONEY_GAIN_MULTIPLIER * money_diff
        elif money_diff < 0:
            reward -= MONEY_SPEND_PENALTY * abs(money_diff)
        
        # Health management (healing is good strategy)
        for i in range(min(len(current_pokemon_hp), len(self.previous_pokemon_hp))):
            if current_pokemon_hp[i] > self.previous_pokemon_hp[i]:
                heal_amount = current_pokemon_hp[i] - self.previous_pokemon_hp[i]
                reward += HEALING_BASE_REWARD + (heal_amount * HEALING_HP_MULTIPLIER)
        
        # ===================================================================
        # BASIC MOVEMENT & TIME
        # ===================================================================
        
        # Simple movement bonus (any position change is action)
        if new_pos_key != prev_pos_key:
            reward += MOVEMENT_BONUS
        
        # Time penalty (encourages efficiency)
        reward -= TIME_PENALTY
        
        # ===================================================================
        # MAP EXPLORATION (General area discovery, not specific locations)
        # ===================================================================
        
        current_map = new_pos_key[2]
        prev_map = prev_pos_key[2]
        
        # Reward for discovering ANY new area (generalizes to any game)
        if current_map != prev_map:
            if current_map not in self.visited_maps:
                # First time visiting this area - good exploration!
                self.visited_maps.add(current_map)
                map_reward = MAP_DISCOVERY_BONUS
                reward += map_reward
                self.reward_components['exploration'] += map_reward
                print(f"  Agent {self.agent_id}: 🗺️  NEW AREA DISCOVERED! ID:{current_map}")
            else:
                # Revisiting a known area - check timing
                steps_since_last = self.steps - self.map_visit_times.get(current_map, 0)
                if steps_since_last > MAP_REVISIT_THRESHOLD:
                    # Legitimate return after exploration
                    revisit_bonus = MAP_REVISIT_LATE_BONUS
                    reward += revisit_bonus
                    self.reward_components['exploration'] += revisit_bonus
                else:
                    # Area flipping exploit - penalize
                    flip_penalty = MAP_REVISIT_QUICK_PENALTY
                    reward -= flip_penalty
                    self.reward_components['penalty'] += flip_penalty
            
            # Update area visit time
            self.map_visit_times[current_map] = self.steps
        
        # === MOVEMENT REWARDS (memory-based, not position-based) ===
        # Movement is now rewarded through the screen hash memory system above
        # This prevents gaming the system by moving between two spots repeatedly
        
        # Small bonus just for taking any movement action
        if new_pos_key != prev_pos_key:
            move_bonus = 0.1
            reward += move_bonus
            self.reward_components['movement'] += move_bonus
        
        # Base time penalty (encourages action)
        time_cost = 0.005
        reward -= time_cost
        self.reward_components['penalty'] += time_cost
        
        # Update previous state tracking
        self.previous_money = current_money
        self.previous_badges = current_badges
        self.previous_item_count = current_item_count
        self.previous_pokemon_count = current_pokemon_count
        self.previous_elite_four_flags = current_elite_four_flags
        self.previous_pokemon_hp = current_pokemon_hp.copy()
        
        return reward
    
    def reset(self):
        """Reset environment."""
        # Log episode data to TensorBoard before resetting
        if self.tensorboard_manager and self.total_episodes > 0:
            self.tensorboard_manager.log_episode(
                self.agent_id, self.total_episodes, 
                self.episode_reward, self.steps, len(self.visited_positions)
            )
            
            # Log game state
            self.tensorboard_manager.log_game_state(
                self.agent_id, self.total_episodes,
                money=self._get_money(),
                badges=bin(self._get_badges()).count('1'),
                pokemon_count=self._get_pokemon_count(),
                items=self._get_item_count(),
                pokedex_owned=self._get_pokedex_owned_count(),
                battles_won=self.battles_won
            )
            
            # Log exploration metrics
            self.tensorboard_manager.log_exploration(
                self.agent_id, self.total_episodes,
                maps_visited=len(self.visited_maps),
                total_states_seen=len(self.state_memory),
                screen_hash_unique=len(set(self.screen_hash_history))
            )
            
            # Log reward breakdown
            self.tensorboard_manager.log_reward_breakdown(
                self.agent_id, self.total_episodes,
                self.reward_components
            )
            
            # Log screen occasionally (every 10 episodes)
            if self.total_episodes % 10 == 0:
                try:
                    screen = self.pyboy.screen.ndarray.copy()
                    self.tensorboard_manager.log_screen(
                        self.agent_id, self.total_episodes, screen
                    )
                except:
                    pass  # Skip if screen capture fails
        
        with open('playable_state.state', 'rb') as f:
            self.pyboy.load_state(f)
        
        time.sleep(0.05)
        
        self.steps = 0
        self.episode_reward = 0
        self.total_episodes += 1
        
        # Reset direction tracking
        self.last_direction = None
        self.direction_consistency_count = 0
        
        # Reset memory systems for new episode
        self.visited_maps.clear()
        self.map_visit_times.clear()
        self.state_memory.clear()
        self.screen_hash_history.clear()
        
        # Initialize previous state tracking
        self.previous_money = self._get_money()
        self.previous_badges = self._get_badges()
        self.previous_item_count = self._get_item_count()
        self.previous_pokemon_count = self._get_pokemon_count()
        self.previous_elite_four_flags = self._get_elite_four_flags()
        self.previous_pokemon_hp = self._get_pokemon_hp()
        self.visited_states.clear()  # Reset visited states for new episode
        
        # Initialize battle and exploration tracking
        self.previous_enemy_hp = self._get_enemy_hp()
        self.previous_current_pokemon_hp = self._get_current_pokemon_battle_hp()
        self.previous_pokedex_owned = self._get_pokedex_owned_count()
        self.in_battle_last_step = self._is_in_battle()
        self.battle_victory_detected = False
        
        self._update_position_tracking()
        
        return self.get_enhanced_state(self.device)
    
    def close(self):
        """Close environment."""
        self.pyboy.stop()

# ============================================================================
# PPO FUNCTIONS
# ============================================================================

def compute_gae(rewards, values, dones, gamma=DISCOUNT_FACTOR, lam=GAE_LAMBDA, device='cpu'):
    """Compute GAE."""
    advantages = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[i + 1]
        
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.FloatTensor(advantages).to(device)
    returns = advantages + torch.FloatTensor(values).to(device)
    
    return advantages, returns

def ppo_update(policy, optimizer, screens, features, actions, old_log_probs, advantages, returns, 
               epochs=PPO_EPOCHS, clip=PPO_CLIP):
    """PPO update with enhanced state."""
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(epochs):
        logits, values = policy(screens, features)
        values = values.squeeze()
        
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = ((returns - values) ** 2).mean()
        entropy = dist.entropy().mean()
        
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
    
    return policy_loss.item(), value_loss.item(), entropy.item()

# ============================================================================
# TRAINING
# ============================================================================

training_interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    global training_interrupted
    print("\n\nTRAINING INTERRUPTED - Saving...")
    training_interrupted = True

def train_agent(agent_id, rom_path, num_episodes, visible, window_pos, 
                progress_queue, screen_capture_manager, shared_models_dir=None, 
                start_episode=0, tensorboard_log_dir=None):
    """Train single agent with PPO and knowledge pooling."""
    global training_interrupted
    
    print(f"\nAgent {agent_id}: Starting PPO training")
    
    # Create TensorBoard manager for this process
    tensorboard_manager = None
    if tensorboard_log_dir:
        tensorboard_manager = TensorBoardManager(log_dir='tensorboard_logs', run_name=tensorboard_log_dir)
    
    # Create environment
    env = PokemonEnv(
        rom_path, agent_id, 
        visible=visible, 
        window_pos=window_pos,
        progress_queue=progress_queue,
        screen_capture_manager=screen_capture_manager if agent_id == 0 else None,
        device=DEVICE,
        tensorboard_manager=tensorboard_manager
    )
    
    # Create policy
    device = DEVICE  # Use detected optimal device
    policy = ActorCritic(action_size=7, device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Initialize policy with dummy forward pass FIRST (before loading)
    try:
        # Create a dummy environment just for initialization
        temp_env = PokemonEnv(rom_path, agent_id, visible=False, device=device)
        dummy_state = temp_env.reset()
        temp_env.close()
        
        if dummy_state is None:
            raise ValueError("Environment reset returned None")
        
        # dummy_state is now a tuple (screen, features)
        if not isinstance(dummy_state, tuple) or len(dummy_state) != 2:
            raise ValueError(f"Expected tuple of (screen, features), got {type(dummy_state)}")
        
        dummy_screen, dummy_features = dummy_state
        
        # Validate tensor shapes
        if not isinstance(dummy_screen, torch.Tensor) or len(dummy_screen.shape) != 4:
            raise ValueError(f"Invalid screen tensor shape: {dummy_screen.shape if isinstance(dummy_screen, torch.Tensor) else type(dummy_screen)}")
        if not isinstance(dummy_features, torch.Tensor):
            raise ValueError(f"Expected features tensor, got {type(dummy_features)}")
        
        with torch.no_grad():
            _ = policy(dummy_screen, dummy_features)  # Initialize model layers
        
        print(f"  Agent {agent_id}: Model initialized successfully with enhanced state features")
        
    except Exception as e:
        print(f"  Agent {agent_id}: Model initialization failed: {e}")
        return
    
    # ALWAYS try to load existing model (resume by default)
    model_path = f'agent_{agent_id}_ppo.pth'
    if os.path.exists(model_path):
        try:
            policy.load_state_dict(torch.load(model_path, map_location=device))
            print(f"  Agent {agent_id}: Loaded existing model from {model_path} (resuming training)")
        except Exception as e:
            print(f"  Agent {agent_id}: Could not load existing model: {e}, starting fresh")
    else:
        print(f"  Agent {agent_id}: No existing model found, starting fresh")
    
    # Create shared model manager if knowledge pooling is enabled
    shared_model_manager = None
    if shared_models_dir:
        shared_model_manager = SharedModelManager(shared_models_dir)
    
    # Training loop
    for episode in range(start_episode, num_episodes):
        if training_interrupted:
            break
        
        # Storage
        states_batch = []
        actions_batch = []
        rewards_batch = []
        values_batch = []
        log_probs_batch = []
        dones_batch = []
        
        # Track losses for this episode
        episode_policy_loss = 0
        episode_value_loss = 0
        episode_entropy = 0
        update_count = 0
        
        # Reset
        state = env.reset()
        done = False
        steps = 0
        
        # Collect experience
        while not done and steps < MAX_STEPS_PER_EPISODE:
            if training_interrupted:
                break
            
            try:
                # Unpack state (screen, features)
                state_screen, state_features = state
                
                # Get action
                with torch.no_grad():
                    logits, value = policy(state_screen, state_features)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                # Step
                next_state, reward, done, info = env.step(action.item())
                
                # Validate next_state
                if next_state is None or not isinstance(next_state, tuple):
                    print(f"  Agent {agent_id}: Invalid next_state: {type(next_state)}")
                    break
                
            except Exception as e:
                print(f"  Agent {agent_id}: Error during step {steps}: {e}")
                break
            
            # Store (store screen and features separately)
            states_batch.append((state_screen, state_features))
            actions_batch.append(action)
            rewards_batch.append(reward)
            values_batch.append(value.item())
            log_probs_batch.append(log_prob)
            dones_batch.append(1.0 if done else 0.0)
            
            state = next_state
            steps += 1
            
            # Update when batch full
            if len(states_batch) >= BATCH_SIZE or done:
                try:
                    advantages, returns = compute_gae(
                        rewards_batch, values_batch, dones_batch, device=device
                    )
                    
                    # Separate screens and features
                    screens_list = [s[0] for s in states_batch]
                    features_list = [s[1] for s in states_batch]
                    
                    screens_tensor = torch.cat(screens_list, dim=0).to(device)
                    features_tensor = torch.cat(features_list, dim=0).to(device)
                    actions_tensor = torch.LongTensor([a.item() for a in actions_batch]).to(device)
                    old_log_probs_tensor = torch.stack(log_probs_batch).to(device)
                    
                    policy_loss, value_loss, entropy = ppo_update(
                        policy, optimizer, screens_tensor, features_tensor, actions_tensor,
                        old_log_probs_tensor, advantages, returns
                    )
                    
                    # Accumulate losses
                    episode_policy_loss += policy_loss
                    episode_value_loss += value_loss
                    episode_entropy += entropy
                    update_count += 1
                    
                except Exception as e:
                    print(f"  Agent {agent_id}: PPO update error: {e}")
                    # Continue without the update
                
                # Clear
                states_batch = []
                actions_batch = []
                rewards_batch = []
                values_batch = []
                log_probs_batch = []
                dones_batch = []
        
        # Print progress every episode
        if not training_interrupted:
            print(f"  Agent {agent_id} Ep {episode}/{num_episodes}: "
                  f"R={env.episode_reward:.1f}, "
                  f"UniquePos={len(env.visited_positions)}, "
                  f"Steps={steps}")
            
            # Log losses to TensorBoard
            if tensorboard_manager and update_count > 0:
                avg_policy_loss = episode_policy_loss / update_count
                avg_value_loss = episode_value_loss / update_count
                avg_entropy = episode_entropy / update_count
                
                tensorboard_manager.log_episode(
                    agent_id, episode, env.episode_reward, steps, len(env.visited_positions),
                    policy_loss=avg_policy_loss, value_loss=avg_value_loss, entropy=avg_entropy
                )
        
        # Save training progress
        if not training_interrupted:
            # Save model after each episode for resume capability
            cpu_state_dict = {}
            for key, tensor in policy.state_dict().items():
                cpu_state_dict[key] = tensor.cpu()
            torch.save(cpu_state_dict, f'agent_{agent_id}_ppo.pth')
            
            # Save training state (this will be called from main process, but we save model here)
        
        # Knowledge pooling at checkpoints
        if shared_model_manager and (episode + 1) % POOLING_FREQUENCY == 0:
            # Save current model for others to use
            shared_model_manager.save_agent_model(agent_id, policy.state_dict(), episode + 1)
            
            # Get knowledge from other agents and mix with current model
            mixed_state = shared_model_manager.pool_knowledge(agent_id, policy.state_dict())
            policy.load_state_dict(mixed_state)
            
            # Log knowledge pooling event
            if tensorboard_manager:
                agents_pooled = len(shared_model_manager.get_available_models(agent_id))
                tensorboard_manager.log_knowledge_pooling(agent_id, episode, agents_pooled)
            
            # Cleanup old models periodically
            if episode > 10:
                shared_model_manager.cleanup_old_models()
    
    # Save final model
    cpu_state_dict = {}
    for key, tensor in policy.state_dict().items():
        cpu_state_dict[key] = tensor.cpu()
    torch.save(cpu_state_dict, f'agent_{agent_id}_ppo.pth')
    
    print(f"\n✓ Agent {agent_id} complete! Unique positions: {len(env.visited_positions)}")
    
    # Close TensorBoard writer
    if tensorboard_manager:
        tensorboard_manager.close_all()
    
    env.close()

# ============================================================================
# PROGRESS MONITOR
# ============================================================================

def monitor_progress(progress_queue, num_agents, tensorboard_log_dir=None):
    """Monitor and display training progress."""
    agent_stats = {i: {} for i in range(num_agents)}
    agent_episodes = {i: 0 for i in range(num_agents)}  # Track completed episodes
    
    # Create TensorBoard manager for this process
    tensorboard_manager = None
    if tensorboard_log_dir:
        tensorboard_manager = TensorBoardManager(log_dir='tensorboard_logs', run_name=tensorboard_log_dir)
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS MONITOR")
    print("="*80)
    print("Receiving updates from agents...\n")
    
    last_print = time.time()
    last_save = time.time()
    last_global_log = time.time()
    
    while True:
        try:
            # Get update (with timeout)
            update = progress_queue.get(timeout=1)
            
            # Update stats
            agent_id = update['agent_id']
            agent_stats[agent_id] = update
            
            # Update episode count if this is a new episode completion
            current_episode = update.get('episode', 0)
            if current_episode > agent_episodes[agent_id]:
                agent_episodes[agent_id] = current_episode
            
            # Save training state every 30 seconds
            if time.time() - last_save > 30:
                save_training_state(agent_episodes)
                last_save = time.time()
            
            # Log global metrics every 60 seconds
            if tensorboard_manager and time.time() - last_global_log > 60:
                if len(agent_stats) > 0:
                    rewards = [s.get('reward', 0) for s in agent_stats.values() if s]
                    steps = [s.get('steps', 0) for s in agent_stats.values() if s]
                    unique_positions = [s.get('unique_positions', 0) for s in agent_stats.values() if s]
                    
                    if rewards:
                        avg_reward = sum(rewards) / len(rewards)
                        avg_steps = sum(steps) / len(steps)
                        total_unique = sum(unique_positions)
                        
                        best_agent_id = max(agent_stats.keys(), key=lambda x: agent_stats[x].get('reward', 0) if agent_stats[x] else 0)
                        best_reward = agent_stats[best_agent_id].get('reward', 0) if agent_stats[best_agent_id] else 0
                        
                        # Use max episode as the global episode counter
                        global_episode = max(agent_episodes.values())
                        
                        tensorboard_manager.log_global_metrics(
                            global_episode, avg_reward, avg_steps, total_unique,
                            best_agent_id, best_reward
                        )
                        
                        # Log system metrics
                        cpu_percent = psutil.cpu_percent()
                        memory_percent = psutil.virtual_memory().percent
                        gpu_memory_mb = 0
                        if torch.cuda.is_available():
                            gpu_memory_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
                        
                        tensorboard_manager.log_system_metrics(
                            global_episode, cpu_percent, memory_percent, gpu_memory_mb
                        )
                
                last_global_log = time.time()
            
            # Print every 10 seconds
            if time.time() - last_print > 5:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Current Status:")
                print("-" * 80)
                for aid in sorted(agent_stats.keys()):
                    if agent_stats[aid]:
                        s = agent_stats[aid]
                        pos = s.get('position', (0, 0, 0))
                        print(f"  Agent {aid}: "
                              f"Ep={s.get('episode', 0):3d}, "
                              f"Steps={s.get('steps', 0):4d}, "
                              f"R={s.get('reward', 0):6.1f}, "
                              f"Pos={s.get('unique_positions', 0):3d}, "
                              f"Map={pos[2]:2d} ({pos[0]},{pos[1]})")
                print("-" * 80)
                last_print = time.time()
                
        except:
            # Timeout - no updates
            continue

# ============================================================================
# TRAINING STATE MANAGEMENT
# ============================================================================

def save_training_state(agent_episodes_completed, training_stats_file='training_state.pkl'):
    """Save the current training state for resume capability."""
    state = {
        'agent_episodes_completed': agent_episodes_completed,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_agents': NUM_AGENTS,
            'episodes_per_agent': EPISODES_PER_AGENT,
            'device': DEVICE
        }
    }
    with open(training_stats_file, 'wb') as f:
        pickle.dump(state, f)

def load_training_state(training_stats_file='training_state.pkl'):
    """Load training state if it exists and is valid."""
    if not os.path.exists(training_stats_file):
        return None
    
    try:
        with open(training_stats_file, 'rb') as f:
            state = pickle.load(f)
        
        # Validate the state matches current configuration
        config = state.get('config', {})
        if (config.get('num_agents') == NUM_AGENTS and 
            config.get('episodes_per_agent') == EPISODES_PER_AGENT and
            config.get('device') == DEVICE):
            return state
        else:
            print("⚠️  Training configuration changed, starting fresh training")
            return None
    except Exception as e:
        print(f"⚠️  Could not load training state: {e}")
        return None

def check_resume_training():
    """Check if we can resume previous training."""
    # Check for training state file
    state = load_training_state()
    if state is None:
        return None, None
    
    agent_episodes_completed = state['agent_episodes_completed']
    
    # Check if all agents have model files
    missing_models = []
    for agent_id in range(NUM_AGENTS):
        model_path = f'agent_{agent_id}_ppo.pth'
        if not os.path.exists(model_path):
            missing_models.append(agent_id)
    
    if missing_models:
        print(f"⚠️  Missing model files for agents: {missing_models}, starting fresh training")
        return None, None
    
    # Check if training was completed
    total_completed = sum(agent_episodes_completed.values())
    total_expected = NUM_AGENTS * EPISODES_PER_AGENT
    
    if total_completed >= total_expected:
        print("✅ Previous training appears complete")
        return None, None
    
    print(f"📋 Found resumable training: {total_completed}/{total_expected} episodes completed")
    return agent_episodes_completed, state['timestamp']

# ============================================================================
# PARALLEL TRAINING
# ============================================================================

def train_swarm_with_monitoring(rom_path, num_agents=NUM_AGENTS, 
                                num_episodes=EPISODES_PER_AGENT):
    """Train multiple agents with live monitoring."""
    global training_interrupted
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n{'#'*80}")
    print(f"PPO SWARM TRAINING: {num_agents} agents")
    print(f"Visible agents: {NUM_VISIBLE_AGENTS}")
    print(f"{'#'*80}")
    print("\nPress Ctrl+C to stop and save")
    print()
    
    # Check for resumable training
    agent_episodes_completed, resume_timestamp = check_resume_training()
    start_episodes = {}
    
    if agent_episodes_completed:
        print(f"🔄 Resuming training from {resume_timestamp}")
        for agent_id in range(num_agents):
            start_episodes[agent_id] = agent_episodes_completed.get(agent_id, 0)
        print(f"   Agent start episodes: {start_episodes}")
    else:
        print("🆕 Starting fresh training")
        for agent_id in range(num_agents):
            start_episodes[agent_id] = 0
    
    # Create shared resources
    manager = Manager()
    progress_queue = manager.Queue()
    screen_capture_manager = ScreenCaptureManager()
    
    # Create TensorBoard run name (to be used by each process independently)
    run_name = f"pokemon_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"📊 TensorBoard logging: tensorboard_logs/{run_name}")
    print(f"   To view: tensorboard --logdir=tensorboard_logs")
    
    # Try to auto-launch TensorBoard
    tensorboard_process = None
    try:
        import subprocess
        import webbrowser
        print("\n🚀 Attempting to auto-launch TensorBoard...")
        tensorboard_process = subprocess.Popen(
            [os.sys.executable, '-m', 'tensorboard.main', '--logdir', 'tensorboard_logs', '--port', '6006'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(2)  # Give it time to start
        if tensorboard_process.poll() is None:
            print("✅ TensorBoard launched successfully!")
            print("   Opening browser to http://localhost:6006...")
            try:
                webbrowser.open('http://localhost:6006')
                print("✅ Browser opened!")
            except:
                print("⚠️  Could not auto-open browser. Visit: http://localhost:6006")
        else:
            print("⚠️  TensorBoard failed to start (may already be running)")
            tensorboard_process = None
    except Exception as e:
        print(f"⚠️  Could not auto-launch TensorBoard: {e}")
        print("   You can manually launch: tensorboard --logdir=tensorboard_logs")
        tensorboard_process = None
    
    # Create shared model manager for knowledge pooling
    shared_models_dir = 'shared_models'
    print(f"🧠 Knowledge pooling enabled: Every {POOLING_FREQUENCY} episodes, α={POOLING_ALPHA}")
    
    # Start progress monitor (pass run_name, not the manager object)
    monitor_proc = mp.Process(target=monitor_progress, args=(progress_queue, num_agents, run_name))
    monitor_proc.start()
    
    # Start training processes
    processes = []
    for agent_id in range(num_agents):
        # Determine if visible
        visible = (agent_id < NUM_VISIBLE_AGENTS)
        window_pos = WINDOW_POSITIONS[agent_id] if visible and agent_id < len(WINDOW_POSITIONS) else (0, 0)
        
        p = mp.Process(
            target=train_agent,
            args=(agent_id, rom_path, num_episodes, visible, window_pos, 
                  progress_queue, screen_capture_manager, shared_models_dir, 
                  start_episodes[agent_id], run_name)
        )
        p.start()
        processes.append(p)
        time.sleep(0.5)
    
    # Track progress for saving state
    agent_progress = {i: start_episodes[i] for i in range(num_agents)}
    
    # Wait for training
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nWaiting for agents to finish...")
        for p in processes:
            p.join(timeout=10)
    
    # Save final training state
    if not training_interrupted:
        # Mark all as completed
        final_episodes = {i: num_episodes for i in range(num_agents)}
        save_training_state(final_episodes)
    
    # Stop monitor
    monitor_proc.terminate()
    monitor_proc.join(timeout=5)
    
    # Stop TensorBoard if we started it
    if tensorboard_process and tensorboard_process.poll() is None:
        print("\n🛑 Stopping TensorBoard...")
        tensorboard_process.terminate()
        try:
            tensorboard_process.wait(timeout=5)
        except:
            tensorboard_process.kill()
    
    # Note: TensorBoard writers are closed automatically in each process
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 TensorBoard logs saved to: tensorboard_logs/{run_name}")
    print(f"   View with: tensorboard --logdir=tensorboard_logs")
    
    # Create best agent model from knowledge pooling
    if os.path.exists(shared_models_dir):
        create_best_agent_model(shared_models_dir)

def create_best_agent_model(shared_models_dir):
    """Create a 'best agent' model by pooling knowledge from all final agent models."""
    print("\n🏆 Creating best agent model from pooled knowledge...")
    
    # Find all final agent models
    final_models = []
    for filename in os.listdir('.'):
        if filename.startswith('agent_') and filename.endswith('_ppo.pth'):
            final_models.append(filename)
    
    if len(final_models) == 0:
        print("❌ No agent models found for pooling")
        return
    
    print(f"📊 Pooling knowledge from {len(final_models)} final agent models...")
    
    # Load all models and average them
    pooled_state = None
    valid_models = 0
    
    for model_path in final_models:
        try:
            agent_state = torch.load(model_path, map_location='cpu')  # Load to CPU first
            valid_models += 1
            
            if pooled_state is None:
                pooled_state = {}
                for key in agent_state.keys():
                    pooled_state[key] = agent_state[key].clone()
            else:
                for key in pooled_state.keys():
                    if key in agent_state:
                        # Move to same device as pooled_state tensors
                        agent_tensor = agent_state[key].to(pooled_state[key].device)
                        pooled_state[key] += agent_tensor
        except Exception as e:
            print(f"⚠️  Failed to load {model_path}: {e}")
            continue
    
    if valid_models > 1:
        # Average all models
        for key in pooled_state.keys():
            pooled_state[key] /= valid_models
        
        # Save the best agent model
        best_model_path = 'best_agent_ppo.pth'
        torch.save(pooled_state, best_model_path)
        print(f"✅ Best agent model saved to {best_model_path} (averaged from {valid_models} agents)")
        return best_model_path
    else:
        print("❌ Need at least 2 valid models for pooling")
        return None

# ============================================================================
# DEMO MODE
# ============================================================================



# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    ROM_PATH = 'pokemon_red.gb'
    
    if not os.path.exists(ROM_PATH):
        print(f"ERROR: {ROM_PATH} not found!")
        exit(1)
    
    print("="*80)
    print("POKEMON PPO TRAINER - AUTO START")
    print("="*80)
    
    if not os.path.exists('playable_state.state'):
        print("\nCreating save state...")
        screen_worked = create_playable_save(ROM_PATH)
        if not screen_worked:
            print("\n⚠ WARNING: Screen didn't change!")
            print("Continuing anyway...")
    
    print(f"\nConfiguration:")
    print(f"  Algorithm: PPO with Enhanced State Space")
    print(f"  Agents: {NUM_AGENTS} ({NUM_VISIBLE_AGENTS} visible)")
    print(f"  Episodes: {EPISODES_PER_AGENT}")
    print(f"  Max steps: {MAX_STEPS_PER_EPISODE}")
    print(f"  Auto-resume: Enabled (will load existing models)")
    print(f"\nFeatures:")
    print(f"  - Live visualization of training progress")
    print(f"  - Console updates every 10 seconds")
    print(f"  - Automatic model saving and resuming")
    print(f"  - Knowledge pooling between agents")
    print(f"  - TensorBoard logging enabled")
    print(f"\n📊 TensorBoard:")
    print(f"  Launch: tensorboard --logdir=tensorboard_logs")
    print(f"  Then open: http://localhost:6006")
    print(f"\nPress Ctrl+C to stop training and save progress")
    
    input("\nPress Enter to start training...")
    
    train_swarm_with_monitoring(ROM_PATH)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\n📊 To view a trained agent, run: python evaluate_model.py")
    print("   (This file should be in your directory for demo mode)")
    print("\nGoodbye!")