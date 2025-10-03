# Pokemon Red RL - Example Usage & Best Practices

## 📚 Complete Usage Examples

### Example 1: First-Time Setup (Beginner)

```bash
# Step 1: Clone or download all files
# Step 2: Place Pokemon Red ROM
cp ~/Downloads/PokemonRed.gb .

# Step 3: Run automated setup
python quickstart.py --install-deps

# This will:
# - Install all dependencies
# - Verify ROM
# - Configure based on your hardware
# - Start training
# - Open TensorBoard automatically
```

### Example 2: Manual Setup (Intermediate)

```bash
# Install dependencies
pip install -r requirements.txt

# Verify system is ready
python setup_check.py

# Start training with custom configuration
python pokemon_red_training.py

# In another terminal, monitor with TensorBoard
tensorboard --logdir ./sessions/pokemon_rl_TIMESTAMP/tensorboard
```

### Example 3: Resume Training (Advanced)

```python
# Edit pokemon_red_training.py, add before model.learn():

# Load checkpoint if exists
checkpoint_path = 'sessions/pokemon_rl_20250103_140530/checkpoints/checkpoint_500000.zip'
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env)
    
    # Important: Reset rollout buffer to match current config
    model.rollout_buffer.reset()
    print("✅ Checkpoint loaded, resuming training")

# Then continue training
model.learn(...)
```

### Example 4: Evaluate Trained Model

```bash
# Visual demo (watch agent play)
python evaluate_model.py \
    --model sessions/pokemon_rl_TIMESTAMP/checkpoints/best_model.zip \
    --mode demo \
    --steps 10000

# Statistical evaluation (collect metrics)
python evaluate_model.py \
    --model sessions/pokemon_rl_TIMESTAMP/checkpoints/best_model.zip \
    --mode eval \
    --episodes 20
```

### Example 5: Generate All Visualizations

```bash
# Single session analysis
python visualize_training.py \
    --session sessions/pokemon_rl_20250103_140530 \
    --output visualizations/run1

# Compare multiple training runs
python visualize_training.py \
    --compare sessions/run1/ sessions/run2/ sessions/run3/ \
    --output visualizations/comparison

# This generates:
# - reward_curve.png
# - episode_length.png
# - exploration.png
# - badges.png
# - pokemon_caught.png
# - dashboard.png (comprehensive overview)
```

### Example 6: Swarm Learning (Multi-Agent)

```bash
# Terminal 1: Start coordinator
python swarm_coordinator.py --agents 4 --sync-freq 100000

# Terminal 2-5: Start individual agents (modify training script to specify agent_id)
# Agent 1
python pokemon_red_training.py --agent-id 0 --swarm-mode

# Agent 2
python pokemon_red_training.py --agent-id 1 --swarm-mode

# ... etc

# Agents will automatically share best weights every 100k steps
```

## 🎯 Best Practices

### Training Best Practices

1. **Start Small, Scale Up**
   ```python
   # First run: Test with fewer environments
   NUM_ENVS = 4
   TOTAL_TIMESTEPS = 1_000_000  # 1M for testing
   
   # Once stable, scale up
   NUM_ENVS = 16
   TOTAL_TIMESTEPS = 50_000_000  # 50M for production
   ```

2. **Monitor Early Progress**
   - Check TensorBoard after 15-30 minutes
   - Episode reward should increase from ~0 to 50+
   - If stuck at 0, check reward function
   - If reward explodes, reduce learning rate

3. **Save Checkpoints Frequently**
   ```python
   SAVE_FREQ = 25_000  # Save every 25k steps
   # Disk space needed: ~50MB per checkpoint
   ```

4. **Use Headless Mode for Speed**
   ```python
   ENV_CONFIG = {
       'headless': True,  # 100x faster training
       'save_screenshots': True  # Still capture milestones
   }
   ```

### Hardware Optimization

**For Systems with 8-16GB RAM:**
```python
NUM_ENVS = 4  # Fewer parallel environments
N_STEPS = 1024  # Smaller rollout buffer
BATCH_SIZE = 256  # Smaller batches
max_steps = 4096  # Shorter episodes
```

**For Systems with 32GB+ RAM:**
```python
NUM_ENVS = 24  # More parallel environments
N_STEPS = 2048  # Standard rollout buffer
BATCH_SIZE = 512  # Standard batches
max_steps = 8192  # Standard episodes
```

**For GPU Acceleration:**
```python
# Install CUDA-enabled PyTorch first
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# Training will automatically use GPU if available
# Check with: torch.cuda.is_available()
```

### Reward Tuning Tips

**If agent gets stuck in menus:**
```python
# Increase menu penalty in _calculate_reward()
menu_penalty = -0.2  # Increased from -0.05
```

**If agent avoids battles:**
```python
# Increase battle rewards
battle_engage_reward = 2.0  # Increased from 0.5
battle_win_reward = 5.0  # Increased from 2.0
```

**If agent dies too often:**
```python
# Increase death penalty
death_penalty = -10.0  # Increased from -5.0
# And add HP management rewards
```

**If agent explores too much without progress:**
```python
# Reduce exploration reward
exploration_reward = 0.5  # Reduced from 1.0
# Increase badge/event rewards
badge_reward = 30.0  # Increased from 20.0
```

### Debugging Common Issues

**Issue: Training very slow (< 10 steps/sec)**
```python
# Solutions:
1. Ensure headless=True
2. Reduce NUM_ENVS
3. Check CPU usage (should be 100% on all cores)
4. Close other applications
5. Increase action_freq to skip more frames
```

**Issue: Out of memory errors**
```python
# Solutions:
1. Reduce NUM_ENVS
2. Reduce N_STEPS (rollout buffer size)
3. Reduce max_steps (episode length)
4. Use CPU instead of GPU if GPU memory limited
5. Close TensorBoard while training
```

**Issue: Agent repeats same actions**
```python
# Solutions:
1. Increase entropy coefficient: ent_coef=0.05
2. Check if stuck penalty is too low
3. Verify action space is correct
4. Add more diverse rewards
```

**Issue: Training unstable (reward crashes)**
```python
# Solutions:
1. Reduce learning rate: 1e-4 instead of 3e-4
2. Increase batch size for stability
3. Use gradient clipping: max_grad_norm=0.5
4. Check for NaN in rewards
```

## 📊 Understanding Training Metrics

### Key Metrics to Watch

1. **Episode Reward**
   - Initial: 0-50 (random exploration)
   - After 1 hour: 50-200 (basic navigation)
   - After 4 hours: 200-500 (competent play)
   - After 12 hours: 500-1500+ (strategic play)

2. **Episode Length**
   - Short episodes (<1000): Agent dying frequently
   - Medium episodes (2000-4000): Normal progress
   - Long episodes (>6000): Agent surviving well

3. **Exploration**
   - Should steadily increase
   - Plateau indicates getting stuck
   - Target: 500+ unique coordinates

4. **Badges**
   - 1st badge: 2-4 hours typical
   - 4 badges: 8-12 hours typical
   - 8 badges: 20-40 hours typical

### Interpreting TensorBoard Graphs

**Reward Curve:**
```
Good: Steady upward trend
Bad: Flat line (not learning)
Bad: Spiky without trend (unstable)
Bad: Sudden drop (catastrophic forgetting)
```

**Policy Entropy:**
```
Good: Starts high (0.5+), gradually decreases
Bad: Drops to 0 quickly (no exploration)
Bad: Stays constant (not learning)
```

**Value Loss:**
```
Good: Decreases over time
Bad: Increases (learning degrading)
Bad: Spikes repeatedly (instability)
```

## 🔧 Advanced Customization

### Custom Reward Components

Add your own reward components in `_calculate_reward()`:

```python
def _calculate_reward(self):
    total_reward = 0.0
    
    # ... existing rewards ...
    
    # Custom: Reward for party diversity
    party_types = self._count_unique_pokemon_types()
    diversity_reward = party_types * 0.5
    total_reward += diversity_reward
    
    # Custom: Reward for money (for buying items)
    current_money = self._read_bcd(MONEY_ADDRESS_1, 3)
    if current_money > self.last_money:
        money_reward = (current_money - self.last_money) * 0.001
        total_reward += money_reward
    self.last_money = current_money
    
    # Custom: Penalty for using same move repeatedly
    if self._check_move_repetition():
        repetition_penalty = -0.2
        total_reward += repetition_penalty
    
    return total_reward
```

### Custom Observation Space

Modify observation to include additional information:

```python
# In __init__
self.observation_space = spaces.Dict({
    'screen': spaces.Box(low=0, high=255, shape=(120, 128, 3), dtype=np.uint8),
    'party_hp': spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32),
    'badges': spaces.MultiBinary(8),
    'position': spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)
})

# In _get_observation
def _get_observation(self):
    return {
        'screen': self._get_screen(),
        'party_hp': self._get_party_hp_normalized(),
        'badges': self._get_badge_array(),
        'position': self._get_position_array()
    }
```

### Custom Action Space

Add more complex actions:

```python
# Combo actions (hold multiple buttons)
self.action_space = spaces.Discrete(15)

self.action_map = [
    'down', 'left', 'right', 'up',  # D-pad
    'a', 'b', 'start', 'select',    # Buttons
    'a+up', 'a+down',                # Combos for menu navigation
    'a+a', 'b+b',                    # Double tap
    'left+a', 'right+a',             # Movement + confirm
    'pass'                           # No-op
]
```

## 🎓 Learning Progression Timeline

### Phase 1: Random Exploration (0-30 min)
- Agent moves randomly
- Discovers basic navigation
- Rewards: 0-50 per episode
- **What to check:** Episode length increasing

### Phase 2: Directed Movement (30 min - 2 hours)
- Agent learns to navigate consistently
- Starts entering buildings
- Engages in some battles
- Rewards: 50-200 per episode
- **What to check:** Exploration count growing

### Phase 3: Battle Competence (2-6 hours)
- Agent fights battles strategically
- Uses items occasionally
- May catch Pokemon
- Rewards: 200-500 per episode
- **What to check:** Win rate improving

### Phase 4: Progress Milestones (6-12 hours)
- Obtains first gym badges
- Navigates between towns
- Manages party health
- Rewards: 500-1000 per episode
- **What to check:** Badge count increasing

### Phase 5: Strategic Play (12-24 hours)
- Consistent badge acquisition
- Catches multiple Pokemon
- Long survival times
- Rewards: 1000-2000+ per episode
- **What to check:** All metrics improving

### Phase 6: Elite Four Attempts (24+ hours)
- Reaches late-game areas
- High-level party
- Complex battle strategies
- **Goal:** Defeat Elite Four

## 📈 Expected Results

Based on the research and implementation:

**With 16 cores, 20GB RAM:**
- First gym badge: 2-4 hours
- Four gym badges: 8-15 hours  
- Eight gym badges: 25-40 hours
- Elite Four ready: 40-60 hours

**Performance metrics:**
- Training speed: 500-1000 steps/second
- Memory usage: 15-25 GB
- Checkpoint size: ~50 MB each
- Total disk usage: 20-50 GB per complete run

## 🏆 Success Criteria

Your agent is learning well if:
- ✅ Episode rewards increase consistently
- ✅ Episode length extends over time
- ✅ Exploration count grows to 500+
- ✅ Badges obtained within expected timeframes
- ✅ TensorBoard shows stable learning curves
- ✅ Screenshots capture key milestones
- ✅ Agent survives longer each episode

## 🚀 Next Steps After Training

1. **Evaluate Performance**
   ```bash
   python evaluate_model.py --model best_model.zip --mode eval --episodes 50
   ```

2. **Generate Publication Graphics**
   ```bash
   python visualize_training.py --session SESSION_DIR --output paper_figures/
   ```

3. **Compare Approaches**
   ```bash
   python visualize_training.py --compare baseline/ tuned/ swarm/
   ```

4. **Deploy Best Model**
   - Save best_model.zip for future use
   - Document hyperparameters used
   - Record performance metrics
   - Share results with community

## 📚 Additional Resources

- **Research Paper:** The implementation is based on proven RL architectures from PWhiddy's Pokemon Red experiments
- **TensorBoard Guide:** Real-time monitoring at http://localhost:6006
- **Community:** Share your results and learn from others
- **Further Reading:** PPO algorithm, reward shaping, curriculum learning

---

**Remember:** RL training is about iteration. Start with defaults, observe behavior, tune rewards, and repeat. The agent learns through millions of trials - be patient and monitor progress!

Good luck training your Pokemon speedrunning champion! 🎮🏆