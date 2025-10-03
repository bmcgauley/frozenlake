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

# ============================================================================
# HYPERPARAMETERS - PPO STYLE
# ============================================================================

NUM_AGENTS = 5
NUM_VISIBLE_AGENTS = 1  # Number of agents to show windows for
EPISODES_PER_AGENT = 20
MAX_STEPS_PER_EPISODE = 10000

# PPO hyperparameters
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 5
PPO_CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

# Batch settings
BATCH_SIZE = 512 # Reduced batch size for quicker updates

# Knowledge pooling settings
POOLING_FREQUENCY = 1  # Share knowledge every N episodes
POOLING_ALPHA = 0.1    # Mixing coefficient for shared knowledge (0.1 = 10% shared, 90% individual)

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

# Window positions for visible agents
WINDOW_POSITIONS = [
    (50, 50),
    (400, 50),
    (750, 50),
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
    """PPO Actor-Critic network."""
    def __init__(self, action_size=9, device='cpu'):
        super(ActorCritic, self).__init__()
        
        self.device = device
        
        # Screen encoder
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.conv_out_size = None
        self.body = None
        self.actor = None
        self.critic = None
        self.action_size = action_size
    
    def _get_conv_output(self, shape):
        """Calculate conv output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            output = self.conv(dummy)
            return int(np.prod(output.size()))
    
    def forward(self, screen):
        # Ensure input is on the correct device
        screen = screen.to(self.device)
        
        # Initialize on first pass
        if self.conv_out_size is None:
            self.conv_out_size = self._get_conv_output(screen.shape[1:])
            
            self.body = nn.Sequential(
                nn.Linear(self.conv_out_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(self.device)
            
            self.actor = nn.Linear(512, self.action_size).to(self.device)
            self.critic = nn.Linear(512, 1).to(self.device)
        
        x = self.conv(screen)
        x = x.view(x.size(0), -1)
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
        filename = f'agent_{agent_id}_ep_{episode}.pth'
        filepath = os.path.join(self.agent_models_dir, filename)
        torch.save(model_state_dict, filepath)
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
                other_state = torch.load(model_path)
                valid_models += 1
                
                for key in pooled_state.keys():
                    if key in other_state:
                        pooled_state[key] += other_state[key]
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
                 progress_queue=None, screen_capture_manager=None):
        self.agent_id = agent_id
        self.progress_queue = progress_queue
        self.screen_capture_manager = screen_capture_manager  # Re-enabled for milestones
        
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
        self.actions = ['up', 'down', 'left', 'right', 'a', 'b', 'start', 'select', 'wait']
        
        # Position tracking
        self.visited_positions = {}
        self.position_history = []
        
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
        # Create a simplified state representation
        state = (x, y, map_id, menu_open)
        return hash(state)
    
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
    
    def get_screen_state(self):
        """Get screen as tensor."""
        screen = self.pyboy.screen.ndarray
        gray = np.dot(screen[...,:3], [0.299, 0.587, 0.114])
        gray = gray / 255.0
        return torch.FloatTensor(gray).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
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
        
        next_state = self.get_screen_state()
        
        info = {
            'position': new_pos_key,
            'unique_positions': len(self.visited_positions),
            'episode_reward': self.episode_reward
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, prev_pos_key, new_pos_key):
        """Calculate comprehensive reward based on game progression."""
        reward = 0
        
        # Get current game state
        current_money = self._get_money()
        current_badges = self._get_badges()
        current_item_count = self._get_item_count()
        current_pokemon_count = self._get_pokemon_count()
        current_elite_four_flags = self._get_elite_four_flags()
        current_pokemon_hp = self._get_pokemon_hp()
        current_state_hash = self._get_game_state_hash()
        menu_open = self._is_menu_open()
        
        # === EXPLORATION PENALTIES ===
        # Penalty for visiting previously seen states (encourages exploration)
        if current_state_hash in self.visited_states:
            reward -= 0.01
        else:
            self.visited_states.add(current_state_hash)
            reward += 0.5  # Small bonus for new states
        
        # Penalty for menu being open (encourages active gameplay)
        if menu_open:
            reward -= 0.1
        
        # === BATTLE AND EXPLORATION REWARDS ===
        # Battle state rewards
        in_battle = self._is_in_battle()
        if in_battle:
            reward += 0.01  # Reward for being in battle
            
            # Damage dealing rewards
            enemy_hp = self._get_enemy_hp()
            if self.previous_enemy_hp > 0 and enemy_hp < self.previous_enemy_hp:
                damage_dealt = self.previous_enemy_hp - enemy_hp
                reward += damage_dealt * 0.1  # Reward proportional to damage dealt
        
        # Battle victory detection
        if self._detect_battle_victory():
            if self._is_wild_battle():
                reward += 15.0  # Reward for winning wild battles
                self._capture_milestone("WILD_BATTLE_VICTORY")
            else:
                reward += 25.0  # Higher reward for winning trainer battles
                self._capture_milestone("TRAINER_BATTLE_VICTORY")
            self.battle_victory_detected = True
        
        # Pokemon capture rewards (with multiplier for new pokedex entries)
        if self._detect_pokemon_capture():
            new_entries = self._get_pokedex_owned_count() - self.previous_pokedex_owned
            reward += 30.0 * new_entries * 1.5  # Multiplied reward for new pokedex entries
            self._capture_milestone("POKEMON_CAPTURE")
        
        # Tall grass exploration bonus
        if self._is_in_tall_grass():
            reward += 0.001  # Small bonus for being in wild areas
        
        # Update battle tracking for next step
        self.in_battle_last_step = in_battle
        self.previous_enemy_hp = self._get_enemy_hp()
        self.previous_current_pokemon_hp = self._get_current_pokemon_battle_hp()
        self.previous_pokedex_owned = self._get_pokedex_owned_count()
        
        # === MAJOR PROGRESSION REWARDS ===
        # Gym badges (major progression)
        badge_diff = bin(current_badges).count('1') - bin(self.previous_badges).count('1')
        if badge_diff > 0:
            reward += 100.0 * badge_diff  # Big reward for gym victories
            self._capture_milestone(f"GYM_BADGE_{bin(current_badges).count('1')}")
        
        # Elite Four progression (ultimate goal)
        elite_four_diff = bin(current_elite_four_flags).count('1') - bin(self.previous_elite_four_flags).count('1')
        if elite_four_diff > 0:
            reward += 500.0 * elite_four_diff  # Massive reward for elite four progress
            self._capture_milestone(f"ELITE_FOUR_{bin(current_elite_four_flags).count('1')}")
        
        # === POKEMON AND ITEM MANAGEMENT ===
        # Pokemon count changes (catching pokemon is good, losing some is slightly bad)
        pokemon_diff = current_pokemon_count - self.previous_pokemon_count
        if pokemon_diff > 0:
            reward += 20.0 * pokemon_diff  # Reward for catching pokemon
        elif pokemon_diff < 0:
            reward -= 2.0 * abs(pokemon_diff)  # Small penalty for losing pokemon (sometimes needed)
        
        # Item count changes (gaining items is good, losing some is slightly bad)
        item_diff = current_item_count - self.previous_item_count
        if item_diff > 0:
            reward += 5.0 * item_diff  # Reward for gaining items
        elif item_diff < 0:
            reward -= 0.5 * abs(item_diff)  # Very small penalty for using/losing items
        
        # === RESOURCE MANAGEMENT ===
        # Money changes (gaining money is good)
        money_diff = current_money - self.previous_money
        if money_diff > 0:
            reward += 0.1 * money_diff  # Small reward proportional to money gained
        elif money_diff < 0:
            reward -= 0.01 * abs(money_diff)  # Very small penalty for spending money
        
        # Pokemon healing detection (reward for healing at pokecenters)
        for i in range(min(len(current_pokemon_hp), len(self.previous_pokemon_hp))):
            if current_pokemon_hp[i] > self.previous_pokemon_hp[i]:
                # Pokemon was healed
                heal_amount = current_pokemon_hp[i] - self.previous_pokemon_hp[i]
                reward += 0.5 + (heal_amount * 0.01)  # Base healing reward + proportional bonus
        
        # === MOVEMENT REWARDS (reduced from original) ===
        # Small reward for moving to new positions
        if new_pos_key != prev_pos_key:
            if new_pos_key not in self.visited_positions:
                reward += 1.0  # Reduced reward for new positions
            else:
                visit_count = self.visited_positions[new_pos_key]
                if visit_count <= 3:
                    reward += 0.2 / visit_count  # Much smaller revisit reward
        
        # Base time penalty (encourages action)
        reward -= 0.005
        
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
        with open('playable_state.state', 'rb') as f:
            self.pyboy.load_state(f)
        
        time.sleep(0.05)
        
        self.steps = 0
        self.episode_reward = 0
        self.total_episodes += 1
        
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
        
        return self.get_screen_state()
    
    def close(self):
        """Close environment."""
        self.pyboy.stop()

# ============================================================================
# PPO FUNCTIONS
# ============================================================================

def compute_gae(rewards, values, dones, gamma=DISCOUNT_FACTOR, lam=GAE_LAMBDA):
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
    
    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    
    return advantages, returns

def ppo_update(policy, optimizer, states, actions, old_log_probs, advantages, returns, 
               epochs=PPO_EPOCHS, clip=PPO_CLIP):
    """PPO update."""
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(epochs):
        logits, values = policy(states)
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
                progress_queue, screen_capture_manager, shared_models_dir=None):
    """Train single agent with PPO and knowledge pooling."""
    global training_interrupted
    
    print(f"\nAgent {agent_id}: Starting PPO training")
    
    # Create environment
    env = PokemonEnv(
        rom_path, agent_id, 
        visible=visible, 
        window_pos=window_pos,
        progress_queue=progress_queue,
        screen_capture_manager=screen_capture_manager if agent_id == 0 else None
    )
    
    # Create policy
    device = 'cpu'  # Use CPU for multiprocessing stability
    policy = ActorCritic(action_size=9, device=device).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # Initialize policy with dummy forward pass - with error handling
    try:
        dummy_state = env.reset()
        if dummy_state is None:
            raise ValueError("Environment reset returned None")
        
        # dummy_state is already a tensor from get_screen_state()
        if not isinstance(dummy_state, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(dummy_state)}")
        
        # Validate tensor shape - should be [1, 1, height, width]
        if len(dummy_state.shape) != 4:
            raise ValueError(f"Invalid screen tensor shape: {dummy_state.shape}")
        
        with torch.no_grad():
            _ = policy(dummy_state)  # Initialize model layers
        
        print(f"  Agent {agent_id}: Model initialized successfully")
        
    except Exception as e:
        print(f"  Agent {agent_id}: Model initialization failed: {e}")
        print(f"  Agent {agent_id}: Dummy state type: {type(dummy_state)}")
        print(f"  Agent {agent_id}: Dummy state shape: {dummy_state.shape if hasattr(dummy_state, 'shape') else 'No shape attr'}")
        env.close()
        return
    
    # Create shared model manager if knowledge pooling is enabled
    shared_model_manager = None
    if shared_models_dir:
        shared_model_manager = SharedModelManager(shared_models_dir)
    
    # Training loop
    for episode in range(num_episodes):
        if training_interrupted:
            break
        
        # Storage
        states_batch = []
        actions_batch = []
        rewards_batch = []
        values_batch = []
        log_probs_batch = []
        dones_batch = []
        
        # Reset
        state = env.reset()
        done = False
        steps = 0
        
        # Collect experience
        while not done and steps < MAX_STEPS_PER_EPISODE:
            if training_interrupted:
                break
            
            try:
                # Get action
                with torch.no_grad():
                    logits, value = policy(state)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                # Step
                next_state, reward, done, info = env.step(action.item())
                
                # Validate next_state
                if next_state is None or not isinstance(next_state, torch.Tensor):
                    print(f"  Agent {agent_id}: Invalid next_state: {type(next_state)}")
                    break
                
            except Exception as e:
                print(f"  Agent {agent_id}: Error during step {steps}: {e}")
                break
            
            # Store
            states_batch.append(state)
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
                        rewards_batch, values_batch, dones_batch
                    )
                    
                    states_tensor = torch.cat(states_batch, dim=0)
                    actions_tensor = torch.LongTensor([a.item() for a in actions_batch])
                    old_log_probs_tensor = torch.stack(log_probs_batch)
                    
                    ppo_update(
                        policy, optimizer, states_tensor, actions_tensor,
                        old_log_probs_tensor, advantages, returns
                    )
                    
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
        
        # Knowledge pooling at checkpoints
        if shared_model_manager and (episode + 1) % POOLING_FREQUENCY == 0:
            # Save current model for others to use
            shared_model_manager.save_agent_model(agent_id, policy.state_dict(), episode + 1)
            
            # Get knowledge from other agents and mix with current model
            mixed_state = shared_model_manager.pool_knowledge(agent_id, policy.state_dict())
            policy.load_state_dict(mixed_state)
            
            # Cleanup old models periodically
            if episode > 10:
                shared_model_manager.cleanup_old_models()
    
    # Save final model
    torch.save(policy.state_dict(), f'agent_{agent_id}_ppo.pth')
    
    print(f"\n✓ Agent {agent_id} complete! Unique positions: {len(env.visited_positions)}")
    
    env.close()

# ============================================================================
# PROGRESS MONITOR
# ============================================================================

def monitor_progress(progress_queue, num_agents):
    """Monitor and display training progress."""
    agent_stats = {i: {} for i in range(num_agents)}
    
    print("\n" + "="*80)
    print("TRAINING PROGRESS MONITOR")
    print("="*80)
    print("Receiving updates from agents...\n")
    
    last_print = time.time()
    
    while True:
        try:
            # Get update (with timeout)
            update = progress_queue.get(timeout=1)
            
            # Update stats
            agent_id = update['agent_id']
            agent_stats[agent_id] = update
            
            # Print every 10 seconds
            if time.time() - last_print > 10:
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
    
    # Create shared resources
    manager = Manager()
    progress_queue = manager.Queue()
    screen_capture_manager = ScreenCaptureManager()
    
    # Create shared model manager for knowledge pooling
    shared_models_dir = 'shared_models'
    print(f"🧠 Knowledge pooling enabled: Every {POOLING_FREQUENCY} episodes, α={POOLING_ALPHA}")
    
    # Start progress monitor
    monitor_proc = mp.Process(target=monitor_progress, args=(progress_queue, num_agents))
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
                  progress_queue, screen_capture_manager, shared_models_dir)
        )
        p.start()
        processes.append(p)
        time.sleep(0.5)
    
    # Wait for training
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nWaiting for agents to finish...")
        for p in processes:
            p.join(timeout=10)
    
    # Stop monitor
    monitor_proc.terminate()
    monitor_proc.join(timeout=5)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    
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
            agent_state = torch.load(model_path)
            valid_models += 1
            
            if pooled_state is None:
                pooled_state = {}
                for key in agent_state.keys():
                    pooled_state[key] = agent_state[key].clone()
            else:
                for key in pooled_state.keys():
                    if key in agent_state:
                        pooled_state[key] += agent_state[key]
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

def demo_agent(agent_id, rom_path, max_steps=5000):
    """Watch trained agent."""
    print(f"\n{'='*80}")
    print(f"DEMO MODE - Agent {agent_id}")
    print(f"{'='*80}\n")

    # Check for best agent model first
    if agent_id == 'best' or agent_id == -1:
        model_path = 'best_agent_ppo.pth'
        if os.path.exists(model_path):
            display_name = "Best Agent (Pooled Knowledge)"
        else:
            print("❌ Best agent model not found! Using Agent 0 instead.")
            model_path = 'agent_0_ppo.pth'
            display_name = "Agent 0"
    else:
        model_path = f'agent_{agent_id}_ppo.pth'
        display_name = f"Agent {agent_id}"
    
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found!")
        return

    print(f"Loading model: {display_name}")

    # Create environment first to get proper screen dimensions
    env = PokemonEnv(rom_path, 0, visible=True, window_pos=(100, 100))
    
    # Initialize policy with proper dimensions
    device = 'cpu'
    policy = ActorCritic(action_size=9, device=device).to(device)
    
    # Initialize the model with a dummy forward pass
    dummy_state = env.reset()
    with torch.no_grad():
        _ = policy(dummy_state)  # This initializes the model layers
    
    # Now load the saved state dict
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    print(f"Loaded PPO policy from {model_path}")
    
    env.pyboy.set_emulation_speed(1)
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print("\nWatching agent...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while steps < max_steps:
            with torch.no_grad():
                logits, _ = policy(state)
                probs = torch.softmax(logits, dim=-1)
                action = probs.argmax()
            
            next_state, reward, done, info = env.step(action.item())
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if steps % 100 == 0:
                pos = info['position']
                print(f"Steps: {steps}, "
                      f"UniquePos: {info['unique_positions']}, "
                      f"R: {total_reward:.1f}, "
                      f"Map: {pos[2]} ({pos[0]},{pos[1]})")
            
            if done:
                print(f"\nEpisode ended at {steps}")
                state = env.reset()
                total_reward = 0
            
            time.sleep(0.02)
    
    except KeyboardInterrupt:
        print("\n\nDemo stopped")
    
    print(f"\nFinal: {steps} steps, {len(env.visited_positions)} unique positions")
    
    env.close()

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
    print("POKEMON PPO TRAINER")
    print("="*80)
    
    if not os.path.exists('playable_state.state'):
        print("\nCreating save state...")
        screen_worked = create_playable_save(ROM_PATH)
        if not screen_worked:
            print("\n⚠ WARNING: Screen didn't change!")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                exit(1)
    
    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("\n1. Train with live monitoring (PPO)")
        print("2. Demo trained agent")
        print("3. Exit")
        
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == '1':
            print(f"\nConfiguration:")
            print(f"  Algorithm: PPO")
            print(f"  Agents: {NUM_AGENTS} ({NUM_VISIBLE_AGENTS} visible)")
            print(f"  Episodes: {EPISODES_PER_AGENT}")
            print(f"  Max steps: {MAX_STEPS_PER_EPISODE}")
            print(f"\nLive console updates every 10 seconds")
            print(f"Screen captures saved to: screen_captures/")
            
            confirm = input("\nStart? (y/n): ").strip().lower()
            if confirm == 'y':
                train_swarm_with_monitoring(ROM_PATH)
        
        elif choice == '2':
            available = [i for i in range(NUM_AGENTS) 
                        if os.path.exists(f'agent_{i}_ppo.pth')]
            
            if not available and not os.path.exists('best_agent_ppo.pth'):
                print("\nNo trained agents!")
                continue
            
            options = []
            if available:
                options.extend(available)
            if os.path.exists('best_agent_ppo.pth'):
                options.append('best')
            
            print(f"\nAvailable: {available}")
            if os.path.exists('best_agent_ppo.pth'):
                print("Special: 'best' (pooled knowledge from all agents)")
            
            agent_input = input("Agent ID (0 or 'best'): ").strip().lower()
            
            if agent_input == 'best' or agent_input == 'b':
                demo_agent('best', ROM_PATH)
            else:
                agent_id = int(agent_input) if agent_input.isdigit() else 0
                if agent_id not in available:
                    print(f"Agent {agent_id} not found!")
                    continue
                demo_agent(agent_id, ROM_PATH)
        
        elif choice == '3':
            print("\nGoodbye!")
            break