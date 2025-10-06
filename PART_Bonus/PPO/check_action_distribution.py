"""
Quick diagnostic to check if the trained model can actually output all action types.
This will show the probability distribution over all 9 actions.
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from evaluate_model import PokemonRedEvalEnv

def check_action_probabilities(model_path, rom_path, num_samples=100):
    """
    Sample the policy network to see action probability distribution.
    
    Args:
        model_path: Path to trained model
        rom_path: Path to Pokemon Red ROM
        num_samples: Number of observations to sample
    """
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION DIAGNOSTIC")
    print("="*80)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    print(f"Loading ROM: {rom_path}")
    env = PokemonRedEvalEnv(rom_path, headless=True, save_screenshots=False)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Sample actions and track probabilities
    action_names = ['down', 'left', 'right', 'up', 'a', 'b', 'start', 'select', 'wait']
    action_counts = np.zeros(9)
    action_probs_sum = np.zeros(9)
    
    print(f"\nSampling {num_samples} observations from policy network...\n")
    
    for i in range(num_samples):
        # Get action probabilities from policy network
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float()
            if hasattr(model.policy, 'features_extractor'):
                features = model.policy.features_extractor(obs_tensor)
            else:
                features = obs_tensor
            
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            action_logits = model.policy.action_net(latent_pi)
            action_probs = torch.softmax(action_logits, dim=-1).numpy()[0]
        
        action_probs_sum += action_probs
        
        # Sample action (stochastic)
        action = np.random.choice(9, p=action_probs)
        action_counts[action] += 1
        
        # Take step in environment
        obs, _, done, _, _ = env.step(action)
        
        if done:
            obs, _ = env.reset()
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  Sampled {i+1}/{num_samples}...", end='\r')
    
    print(f"  Sampled {num_samples}/{num_samples}... Done!\n")
    
    # Calculate statistics
    action_probs_mean = action_probs_sum / num_samples
    action_freq = action_counts / num_samples
    
    # Print results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print("\nAverage Policy Probabilities (what the network thinks):")
    print("-" * 60)
    for i, name in enumerate(action_names):
        bar_length = int(action_probs_mean[i] * 50)
        bar = "█" * bar_length
        print(f"  {name:8s} | {action_probs_mean[i]*100:5.2f}% {bar}")
    
    print("\nActual Action Frequency (when sampling stochastically):")
    print("-" * 60)
    for i, name in enumerate(action_names):
        bar_length = int(action_freq[i] * 50)
        bar = "█" * bar_length
        print(f"  {name:8s} | {action_freq[i]*100:5.2f}% {bar}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    # Check if all actions are possible
    zero_prob_actions = [action_names[i] for i in range(9) if action_probs_mean[i] < 0.01]
    
    if len(zero_prob_actions) == 0:
        print("\n✅ GOOD: All actions have non-zero probability")
        print("   The model CAN output all action types.")
    else:
        print(f"\n⚠️  WARNING: These actions have <1% probability:")
        print(f"   {', '.join(zero_prob_actions)}")
        print("   The model is heavily biased toward certain actions.")
    
    # Check entropy
    entropy = -np.sum(action_probs_mean * np.log(action_probs_mean + 1e-10))
    max_entropy = np.log(9)  # Uniform distribution entropy
    normalized_entropy = entropy / max_entropy
    
    print(f"\nPolicy Entropy: {entropy:.3f} / {max_entropy:.3f} ({normalized_entropy*100:.1f}% of max)")
    if normalized_entropy < 0.3:
        print("   → Very deterministic policy (low exploration)")
    elif normalized_entropy < 0.6:
        print("   → Moderately deterministic policy")
    else:
        print("   → High entropy policy (lots of exploration)")
    
    print("\n" + "="*80)
    
    env.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check action distribution of trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--rom', type=str, default='PokemonRed.gb',
                       help='Path to Pokemon Red ROM')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of observations to sample')
    
    args = parser.parse_args()
    
    check_action_probabilities(args.model, args.rom, args.samples)
