# ANTI-COLLAPSE TRAINING CONFIGURATION

# Hyperparameters specifically tuned to prevent policy collapse
ANTI_COLLAPSE_CONFIG = {
    # PPO Hyperparameters
    'learning_rate': 0.0005,       # Lower learning rate - more stable learning
    'n_steps': 512,                # Longer rollouts - better exploration
    'batch_size': 256,             # Smaller batches relative to n_steps
    'n_epochs': 3,                 # Fewer epochs - prevent overfitting to bad patterns
    'gamma': 0.995,                # Higher gamma - care about long-term rewards
    'gae_lambda': 0.95,            # High GAE - better advantage estimation
    'clip_range': 0.1,             # Smaller clip range - conservative updates
    'clip_range_vf': 0.2,          # Value function clipping
    'ent_coef': 0.3,               # HIGH initial entropy - will be scheduled down
    'vf_coef': 0.8,                # Higher value function coefficient
    'max_grad_norm': 0.3,          # Smaller gradient clipping
    'target_kl': 0.02,             # KL divergence limit to prevent large policy changes
    
    # Environment Configuration  
    'num_envs': 8,                 # More parallel environments for diversity
    'action_freq': 32,             # Slightly longer action duration
    'max_episode_steps': 10000,    # Longer episodes
    
    # Training Schedule
    'total_timesteps': 200000,     # More training time
    'save_freq': 10000,            # More frequent saves
    
    # Policy Network Architecture
    'policy_kwargs': {
        'net_arch': [256, 256],    # Larger network
        'activation_fn': 'tanh',   # Tanh activation (helps with stability)
        'features_extractor_kwargs': {
            'features_dim': 512
        }
    }
}

def create_anti_collapse_model(env, tensorboard_log=None):
    """
    Create PPO model with anti-collapse configuration
    """
    from stable_baselines3 import PPO
    import torch.nn as nn
    
    return PPO(
        'CnnPolicy',
        env,
        learning_rate=ANTI_COLLAPSE_CONFIG['learning_rate'],
        n_steps=ANTI_COLLAPSE_CONFIG['n_steps'],
        batch_size=ANTI_COLLAPSE_CONFIG['batch_size'],
        n_epochs=ANTI_COLLAPSE_CONFIG['n_epochs'],
        gamma=ANTI_COLLAPSE_CONFIG['gamma'],
        gae_lambda=ANTI_COLLAPSE_CONFIG['gae_lambda'],
        clip_range=ANTI_COLLAPSE_CONFIG['clip_range'],
        clip_range_vf=ANTI_COLLAPSE_CONFIG['clip_range_vf'],
        ent_coef=ANTI_COLLAPSE_CONFIG['ent_coef'],
        vf_coef=ANTI_COLLAPSE_CONFIG['vf_coef'],
        max_grad_norm=ANTI_COLLAPSE_CONFIG['max_grad_norm'],
        target_kl=ANTI_COLLAPSE_CONFIG['target_kl'],
        tensorboard_log=tensorboard_log,
        policy_kwargs=ANTI_COLLAPSE_CONFIG['policy_kwargs'],
        verbose=1,
        device='auto'
    )