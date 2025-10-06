# CNN Input Debugging System

## Overview

This system allows you to visualize exactly what the Pokemon Red RL model "sees" - the processed 3-frame stacked grayscale observations that go into the CNN policy. This is incredibly valuable for debugging and understanding agent behavior.

## What Gets Visualized

The CNN receives observations with this structure:
- **Shape**: (36, 40, 3) - Height × Width × Channels  
- **3 Stacked Frames**: Current frame, previous frame, frame before that
- **Grayscale**: Each channel is a grayscale frame (0-255)
- **Reduced Resolution**: 4x smaller than Game Boy screen (144×160 → 36×40)
- **Status Bars**: Simple overlaid bars showing HP, levels, exploration progress

## How It Works

### 1. During Training
When `debug_cnn_input=True` is set, the environment:
- Captures the exact observation sent to the CNN every N steps
- Creates side-by-side visualization of the 3 stacked frames
- Overlays debug information (step count, action, reward, game state)
- Saves both the visualization image and raw numpy data

### 2. File Structure
```
sessions/pokemon_rl_20241005_120000/
├── cnn_debug/                    # Main environment
│   ├── cnn_input_000050.png     # Visualization images
│   ├── cnn_input_000100.png
│   ├── cnn_data_000050.npy      # Raw observation data
│   └── cnn_data_000100.npy
├── cnn_debug_env_1/              # Environment 1 (if using multiple)
└── cnn_debug_env_2/              # Environment 2
```

## Configuration

### Enable CNN Debugging
```python
ENV_CONFIG = {
    # ... other config ...
    'debug_cnn_input': True,      # Enable debugging
    'cnn_save_frequency': 50,     # Save every 50 steps
}
```

### Frequency Options
- `cnn_save_frequency`: How often to save debug frames
  - `10`: Very frequent (good for short tests)
  - `50`: Moderate (good for training sessions)
  - `100`: Less frequent (reduces file count)

## Analysis Tools

### 1. Basic Analysis
```bash
# Analyze the latest training session
python cnn_debug_analyzer.py

# Analyze a specific session
python cnn_debug_analyzer.py sessions/pokemon_rl_20241005_120000
```

### 2. Generated Outputs
The analyzer creates:
- **`model_perception_timeline.mp4`**: Video showing model's perception over time
- **`temporal_analysis.png`**: Charts showing frame differences and movement
- **`analysis_report.json`**: Detailed statistics and metrics

### 3. What You Can Learn

**Frame Differences**: How much the model's input changes between steps
- High differences = lots of movement/change
- Low differences = static/repetitive areas

**Movement Patterns**: Detection of character movement vs static screens
- Useful for identifying stuck behavior
- Shows exploration vs repetitive actions

**Temporal Evolution**: How perception changes during training
- Early training: Random, chaotic patterns
- Later training: More purposeful movement patterns

## Debug Image Format

Each debug image shows:
```
┌─────────────┬─────────────┬─────────────┐
│   Frame -2  │   Frame -1  │   Current   │  ← 3 stacked frames
│  (oldest)   │  (middle)   │  (newest)   │
└─────────────┴─────────────┴─────────────┘
Step: 001050          Action: RIGHT        
Reward: 1.500         Levels: 8           
Unique Screens: 45    HP Bars: ███████    
```

## Common Use Cases

### 1. Debugging Stuck Behavior
If your agent gets stuck, look at the debug frames:
- Are the 3 frames identical? (No movement detected)
- Are status bars changing? (Game progress happening)
- Is the agent trying different actions?

### 2. Understanding Exploration
- Watch the video to see exploration patterns
- Check if the agent returns to previously seen areas
- Identify areas the agent avoids or prefers

### 3. Verifying Input Processing
- Ensure frames are properly stacked
- Check that status bars are visible
- Verify resolution reduction looks correct

### 4. Action-Observation Relationships
- See what the agent "saw" when it took specific actions
- Understand decision patterns in different game areas
- Identify if certain visual patterns trigger specific behaviors

## Performance Considerations

### File Size
- Each debug image: ~50KB
- Each raw data file: ~4KB
- 1000 steps at frequency=50: ~1MB total

### Training Speed
- Minimal impact when frequency ≥ 50
- Some overhead from PIL image processing
- Disable for production training runs

## Tips for Effective Analysis

### 1. Start Small
```python
# For initial testing
'cnn_save_frequency': 10,  # High frequency
'max_steps': 1000,         # Short episodes
```

### 2. Focus on Key Moments
- Increase frequency during interesting segments
- Look for moments when reward spikes occur
- Analyze periods of rapid exploration

### 3. Compare Sessions
- Debug frames from different training stages
- Compare successful vs unsuccessful runs
- Identify what changed in agent perception

## Example Workflow

1. **Enable debugging** in your training config
2. **Run training** for a few thousand steps
3. **Create videos** with `cnn_debug_analyzer.py`
4. **Watch the timeline** to understand agent behavior
5. **Check specific frames** when interesting events occur
6. **Analyze patterns** in the temporal charts

## Troubleshooting

### No Debug Files Created
- Check `debug_cnn_input=True` is set
- Verify the environment is taking steps
- Look for error messages during training

### Large File Sizes
- Increase `cnn_save_frequency` value
- Reduce training duration for testing
- Clean up old debug files regularly

### Video Creation Fails
- Install OpenCV: `pip install opencv-python`
- Check that debug images exist
- Verify sufficient disk space

## Advanced Analysis

### Custom Analysis Scripts
You can load the raw `.npy` files for custom analysis:
```python
import numpy as np

# Load observation data
obs = np.load('cnn_data_001000.npy')

# Analyze frame differences
frame1, frame2, frame3 = obs[:,:,0], obs[:,:,1], obs[:,:,2]
movement = np.mean(np.abs(frame3 - frame2))

# Check status bar regions
status_region = obs[2:10, 32:40, :]  # Status bar area
```

This CNN debugging system gives you unprecedented insight into your Pokemon Red RL agent's "mind" and is essential for understanding and improving training performance!