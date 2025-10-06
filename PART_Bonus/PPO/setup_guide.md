# Pokemon Red Reinforcement Learning Speedrunner

A comprehensive RL implementation that trains an agent to speedrun Pokemon Red from start to Elite Four completion using PPO (Proximal Policy Optimization).

## 🎯 Features

- **Comprehensive Reward Shaping**: Rewards exploration, battle engagement, damage dealt, gym badges, Pokemon catching, and penalizes deaths, getting stuck, and menu dwelling
- **Parallel Training**: Utilizes all CPU cores for fast training
- **Milestone Screenshots**: Automatically captures screenshots at major achievements
- **Swarm Learning**: Saves model weights for sharing progress across agents
- **Real-time Visualization**: TensorBoard integration for live training metrics
- **Evaluation Tools**: Demo mode and statistical evaluation
- **Training Analysis**: Generate publication-quality charts and graphs

## 📋 Requirements

### System Requirements
- **CPU**: 16+ cores recommended (minimum 8)
- **RAM**: 20-32GB
- **Storage**: 10-50GB for training sessions
- **OS**: Linux, macOS, or Windows with Python 3.10+

### Software Dependencies

```bash
# Core dependencies
pip install torch torchvision  # PyTorch
pip install gymnasium          # RL environment interface
pip install stable-baselines3  # PPO implementation
pip install pyboy              # Game Boy emulator

# Visualization and analysis
pip install matplotlib seaborn pandas
pip install scikit-image Pillow
pip install tensorboard

# Optional: For CUDA acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ROM Requirements

You need a legally obtained Pokemon Red ROM file:
- **Filename**: `PokemonRed.gb`
- **Size**: Exactly 1,048,576 bytes (1MB)
- **SHA1**: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`
- **Location**: Same directory as training script

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv pokemon_rl_env
source pokemon_rl_env/bin/activate  # On Windows: pokemon_rl_env\Scripts\activate

# Install all dependencies
pip install torch gymnasium stable-baselines3 pyboy matplotlib seaborn pandas scikit-image Pillow tensorboard
```

### 2. Prepare ROM

```bash
# Place your Pokemon Red ROM in the project directory
cp /path/to/your/PokemonRed.gb .

# Verify ROM integrity
shasum PokemonRed.gb
# Should output: ea9bcae617fdf159b045185467ae58b2e4a48b9a
```

### 3. Start Training

```bash
# Run main training script
python pokemon_red_training.py
```

Training will:
- Create a session directory in `./sessions/`
- Start 16 parallel environments (adjust based on your CPU)
- Save checkpoints every 50,000 steps
- Capture screenshots at major milestones
- Log metrics to TensorBoard

### 4. Monitor Training

Open a new terminal and start TensorBoard:

```bash
tensorboard --logdir ./sessions/pokemon_rl_TIMESTAMP/tensorboard
```

Then open your browser to `http://localhost:6006` to view:
- Real-time reward curves
- Episode length progression
- Custom metrics (badges, exploration, etc.)

### 5. Evaluate Trained Model

After training completes (or during training with checkpoints):

```bash
# Visual demo mode (shows agent playing)
python evaluate_model.py --model sessions/pokemon_rl_TIMESTAMP/checkpoints/best_model.zip --mode demo

# Statistical evaluation (runs multiple episodes)
python evaluate_model.py --model sessions/pokemon_rl_TIMESTAMP/checkpoints/best_model.zip --mode eval --episodes 10
```

### 6. Generate Visualizations

```bash
# Create all training charts
python visualize_training.py --session sessions/pokemon_rl_TIMESTAMP --output visualizations/

# Compare multiple training runs
python visualize_training.py --compare session1/ session2/ session3/ --output comparison/
```

## 📊 Understanding the Reward System

### Positive Rewards

| Event | Reward | Purpose |
|-------|--------|---------|
| New coordinate explored | +1.0 | Encourages exploration |
| Battle engaged | +0.5 | Promotes battling |
| Damage dealt to opponent | +0.1 per HP | Rewards combat effectiveness |
| Battle victory | +2.0 | Major combat success |
| Pokemon caught | +5.0 | Collection progress |
| Gym badge obtained | +20.0 | Major milestone achievement |

### Negative Penalties

| Event | Penalty | Purpose |
|-------|---------|---------|
| Stuck in same position (>10 steps) | -0.1 per step | Prevents getting stuck |
| In menu too long (>20 steps) | -0.05 per step | Discourages menu dwelling |
| All Pokemon fainted (death) | -5.0 | Major failure penalty |

### Reward Philosophy

The reward system is designed to encourage **speedrunning behavior**:
1. Exploration rewards push the agent to discover new areas
2. Battle rewards ensure the agent doesn't avoid combat
3. Badge rewards create clear hierarchical objectives
4. Penalties prevent degenerate behaviors (getting stuck, menu spam)

## 🎮 File Structure

```
pokemon-red-rl/
├── pokemon_red_training.py      # Main training script
├── evaluate_model.py            # Model evaluation and demo
├── visualize_training.py        # Training visualization
├── PokemonRed.gb               # Pokemon Red ROM (you provide)
├── sessions/                    # Training sessions
│   └── pokemon_rl_TIMESTAMP/
│       ├── checkpoints/         # Model weights
│       ├── screenshots/         # Milestone screenshots
│       ├── tensorboard/         # TensorBoard logs
│       └── training_history.json
└── visualizations/              # Generated charts
```

## 🔧 Configuration

### Adjusting Training Parameters

Edit the `pokemon_red_training.py` file to modify:

```python
# Number of parallel environments (line ~850)
NUM_ENVS = min(16, os.cpu_count())  # Reduce if running out of RAM

# Total training duration (line ~851)
TOTAL_TIMESTEPS = 10_000_000  # Increase for longer training

# Checkpoint frequency (line ~852)
SAVE_FREQ = 50_000  # Save more/less frequently

# Learning rate (line ~853)
LEARNING_RATE = 3e-4  # Adjust for faster/slower learning

# Episode length (line ~855)
max_steps = 8192  # Longer episodes = more memory but better learning
```

### Hardware-Specific Tuning

**Low RAM systems (8-16GB):**
```python
NUM_ENVS = 4  # Fewer parallel environments
max_steps = 4096  # Shorter episodes
```

**High-end systems (32GB+, 32+ cores):**
```python
NUM_ENVS = 32  # More parallel environments
N_STEPS = 4096  # Larger batch collection
BATCH_SIZE = 1024  # Larger batch size
```

## 📈 Expected Training Progress

Based on research, here's what to expect:

### Hour 1-2: Initial Learning
- Agent learns basic movement
- Starts exploring first areas
- May get stuck frequently
- Reward: ~50-100 per episode

### Hour 3-6: Competent Navigation
- Reliably navigates towns and routes
- Starts engaging in battles
- May obtain first gym badge
- Reward: ~100-300 per episode

### Hour 8-12: Progressive Play
- Obtains multiple gym badges
- Catches Pokemon regularly
- Strategic battle behavior
- Reward: ~300-800 per episode

### Hour 12-24: Advanced Play
- Consistent badge acquisition
- Long survival times
- May reach Elite Four
- Reward: ~800-2000+ per episode

## 🐛 Troubleshooting

### Issue: PyBoy installation fails

**Solution:**
```bash
# Install system dependencies (Linux)
sudo apt-get install libsdl2-dev

# Install system dependencies (macOS)
brew install sdl2

# Update PyBoy
pip install --upgrade pyboy
```

### Issue: CUDA out of memory

**Solution:**
```python
# Use CPU instead
device = 'cpu'

# Or reduce batch size
BATCH_SIZE = 256
```

### Issue: Training too slow

**Solution:**
1. Reduce NUM_ENVS if using too much RAM
2. Ensure `headless=True` in ENV_CONFIG
3. Close TensorBoard while training
4. Use smaller `max_steps`

### Issue: Agent gets stuck repeatedly

**Solution:**
1. Increase stuck penalty: `stuck_penalty = -0.5`
2. Reduce episode length to reset more frequently
3. Add more exploration reward: `exploration_reward = 2.0`

### Issue: ROM not found error

**Solution:**
```bash
# Verify ROM is in correct location
ls -lh PokemonRed.gb

# Verify ROM integrity
shasum PokemonRed.gb

# Update ROM_PATH in script if needed
ROM_PATH = '/full/path/to/PokemonRed.gb'
```

## 📚 Advanced Usage

### Resuming Training from Checkpoint

```python
# In pokemon_red_training.py, add before model.learn():
checkpoint_path = 'sessions/pokemon_rl_TIMESTAMP/checkpoints/checkpoint_500000.zip'
if os.path.exists(checkpoint_path):
    model = PPO.load(checkpoint_path, env=env)
    print(f"Resumed from: {checkpoint_path}")
```

### Custom Reward Function

Modify `_calculate_reward()` in `PokemonRedEnv` class:

```python
def _calculate_reward(self):
    total_reward = 0.0
    
    # Add your custom reward components here
    # Example: Reward for time efficiency
    time_bonus = 1.0 / (self.current_step + 1)
    total_reward += time_bonus
    
    # ... rest of reward calculation
    return total_reward
```

### Swarm Learning Setup

Share model weights across multiple training instances:

```python
# On agent 1, save frequently
model.save('shared_weights/agent_1_latest.zip')

# On agent 2, load periodically
if os.path.exists('shared_weights/agent_1_latest.zip'):
    model = PPO.load('shared_weights/agent_1_latest.zip', env=env)
```

## 🎯 Milestones to Watch For

The agent will automatically capture screenshots at these milestones:

1. ✅ First battle won
2. 🏅 Boulder Badge (Brock defeated)
3. 🏅 Cascade Badge (Misty defeated)
4. 🏅 Thunder Badge (Lt. Surge defeated)
5. 🏅 Rainbow Badge (Erika defeated)
6. 🏅 Soul Badge (Koga defeated)
7. 🏅 Marsh Badge (Sabrina defeated)
8. 🏅 Volcano Badge (Blaine defeated)
9. 🏅 Earth Badge (Giovanni defeated)
10. 👑 Elite Four entered
11. 🏆 Pokemon Champion!

Screenshots are saved in: `sessions/SESSION_NAME/screenshots/`

## 📊 Performance Metrics

The training system tracks:

- **Episode Reward**: Cumulative reward per episode
- **Episode Length**: Steps until termination
- **Exploration**: Unique coordinates discovered
- **Badges**: Gym badges obtained (0-8)
- **Pokemon Caught**: Number in Pokedex
- **Battles Won/Lost**: Combat statistics
- **Deaths**: Number of party wipes

All metrics are logged to TensorBoard and `training_history.json`.

## 🤝 Contributing & Customization

This implementation is designed for educational purposes and custom modifications:

1. **Custom Games**: Adapt the environment wrapper for other Game Boy games
2. **Different Algorithms**: Replace PPO with A3C, DQN, or others
3. **Reward Engineering**: Experiment with different reward structures
4. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.

## 📝 Citation

If you use this implementation in your research or projects, please cite:

```
Pokemon Red RL Speedrunner
Based on PWhiddy/PokemonRedExperiments architecture
Implementation: Custom PPO-based agent with comprehensive reward shaping
```

## ⚠️ Important Notes

1. **ROM Legality**: Only use ROMs you legally own. This project does not provide ROMs.
2. **Training Time**: Reaching Elite Four may take 12-24+ hours of training
3. **Hardware**: More cores = faster training. 16+ cores strongly recommended.
4. **Checkpoints**: Always save checkpoints frequently to avoid losing progress
5. **Monitoring**: Check TensorBoard regularly to catch training issues early

## 🎓 Learning Resources

To understand the RL concepts used:

- **PPO Algorithm**: [Proximal Policy Optimization paper](https://arxiv.org/abs/1707.06347)
- **Gymnasium Docs**: [farama.org/gymnasium](https://gymnasium.farama.foundation/)
- **Stable-Baselines3**: [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/)
- **PyBoy**: [github.com/Baekalfen/PyBoy](https://github.com/Baekalfen/PyBoy)

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Review TensorBoard logs for training anomalies
3. Verify all dependencies are correctly installed
4. Ensure ROM file is valid and in correct location

---

**Good luck training your Pokemon Red speedrunning agent!** 🎮🚀

Remember: The agent learns through trial and error. Early episodes will look random, but after several hours, you'll see meaningful progress toward game completion!