# 🔄 Model Continuation Guide

## Overview
The Pokemon Red training system now supports continuing training from previously saved checkpoints, allowing you to resume training without losing progress.

## Key Features

### ✅ Automatic Checkpoint Saving
- Models are automatically saved every 5,000 timesteps during training
- Saved to `sessions/session_[timestamp]/checkpoints/ppo_pokemon_[steps].zip`
- Latest model is always available as `ppo_pokemon_latest.zip`

### ✅ Multiple Continuation Methods

#### 1. Command Line Interface
```bash
# Start fresh training
python pokemon_red_training.py --train

# Continue from latest checkpoint in current session
python pokemon_red_training.py --continue

# Continue from specific model file
python pokemon_red_training.py --continue path/to/model.zip

# Show help
python pokemon_red_training.py --help
```

#### 2. Interactive Menu
Run `python pokemon_red_training.py` and select:
- **Option 1**: Start New Training Session
- **Option 2**: Continue Training from Checkpoint
  - Sub-option 1: Auto-find latest checkpoint
  - Sub-option 2: Browse and select specific checkpoint

## How It Works

### 🔧 Technical Details
- **Model Persistence**: Complete model state (weights, optimizer, hyperparameters) is saved
- **Training Continuation**: Training resumes with the exact same learning rate and optimizer state
- **Environment Reset**: New environments are created, but model knowledge is preserved
- **Timestep Tracking**: Training continues from the saved timestep count

### 🧠 Agent Knowledge Sharing
**Important**: Individual agents in parallel environments do NOT share knowledge during training. Each environment runs independently. However:
- ✅ The shared model learns from experiences across all environments
- ✅ When you continue training, the model retains ALL learned knowledge
- ✅ Experience replay and gradient updates combine learnings from all parallel agents

## Best Practices

### 📁 Checkpoint Management
1. **Automatic Cleanup**: Old checkpoints are kept for safety
2. **Session Organization**: Each training session gets its own folder
3. **Backup Important Models**: Copy successful models to a safe location

### 🎯 When to Continue Training
- **Good Performance**: Model is learning but hasn't reached your goals
- **Interrupted Training**: Training was stopped due to time/resource constraints
- **Hyperparameter Adjustment**: Want to continue with modified settings (learning rate, etc.)
- **Extended Training**: Need more timesteps for complex behaviors

### 🚫 When to Start Fresh
- **Poor Performance**: Model has learned bad behaviors or got stuck
- **Major Changes**: Significantly different reward system or environment
- **Architecture Changes**: Different CNN structure or hyperparameters
- **Research Comparison**: Baseline comparison needed

## Example Workflow

### 1. Initial Training
```bash
python pokemon_red_training.py --train
# Trains for 1,000,000 timesteps, saves checkpoints every 5,000 steps
```

### 2. Continue Training
```bash
# Auto-continue from latest
python pokemon_red_training.py --continue

# Or continue from specific checkpoint
python pokemon_red_training.py --continue sessions/session_20241220_143022/checkpoints/ppo_pokemon_50000.zip
```

### 3. Monitor Progress
- Check TensorBoard: `tensorboard --logdir sessions/[session]/tensorboard`
- View CNN debug visualizations in session folder
- Monitor reward trends and learning curves

## Troubleshooting

### ❌ "No checkpoints found"
- Check that training has run long enough to create checkpoints (>5,000 steps)
- Verify session folder structure exists
- Make sure you're in the correct directory

### ❌ "Model loading failed"
- Ensure checkpoint file is not corrupted
- Check file permissions
- Verify stable-baselines3 version compatibility

### ❌ "Environment mismatch"
- Model was trained with different environment settings
- Check observation space dimensions (should be 36x40x3)
- Verify action space is discrete with 9 actions

## Advanced Usage

### Custom Continuation Scripts
```python
from pokemon_red_training import main

# Continue with custom parameters
main(continue_training=True, model_path="path/to/model.zip")
```

### Checkpoint Analysis
```python
from stable_baselines3 import PPO

# Load model for analysis
model = PPO.load("path/to/checkpoint.zip")
print(f"Training timesteps: {model.num_timesteps}")
print(f"Learning rate: {model.learning_rate}")
```

## Monitoring

### 📊 TensorBoard Metrics
When continuing training, look for:
- **Smooth transitions**: No sudden jumps in learning curves
- **Continued improvement**: Reward trends should continue upward
- **Stable entropy**: Should gradually decrease as policy becomes more confident

### 🎮 CNN Debug Visualization
- Compare before/after continuation to see learning progression
- Monitor frame attention patterns
- Check for consistent game state recognition

---

*This system allows you to iteratively improve your Pokemon Red AI over multiple sessions while preserving all learned knowledge.*