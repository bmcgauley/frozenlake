# ENHANCED ENTROPY MANAGEMENT - Prevents Policy Collapse

def create_enhanced_entropy_scheduler():
    """
    Enhanced entropy scheduling that prevents policy collapse
    """
    class EnhancedEntropyScheduler(BaseCallback):
        def __init__(self, 
                     initial_ent_coef=0.3,      # Higher starting entropy
                     final_ent_coef=0.1,        # Higher final entropy  
                     total_timesteps=100000,
                     min_entropy_threshold=1.0): # Stop decay if entropy gets too low
            super().__init__()
            self.initial_ent_coef = initial_ent_coef
            self.final_ent_coef = final_ent_coef
            self.total_timesteps = total_timesteps
            self.min_entropy_threshold = min_entropy_threshold
            self.action_counts = {i: 0 for i in range(9)}
            
        def _on_step(self):
            # Calculate current progress
            progress = min(self.num_timesteps / self.total_timesteps, 1.0)
            
            # Non-linear decay (slower at the beginning)
            decay_factor = progress ** 0.5  # Square root decay
            current_ent_coef = self.initial_ent_coef - (self.initial_ent_coef - self.final_ent_coef) * decay_factor
            
            # Monitor action distribution to detect collapse
            if hasattr(self.model, 'env') and hasattr(self.model.env, 'get_attr'):
                try:
                    # Get recent actions from environments
                    recent_actions = []
                    for env_idx in range(self.model.env.num_envs):
                        env_actions = self.model.env.get_attr('action_history', indices=[env_idx])[0]
                        if env_actions:
                            recent_actions.extend(list(env_actions)[-50:])  # Last 50 actions per env
                    
                    if len(recent_actions) > 100:  # Enough data to analyze
                        # Count action distribution
                        for action in recent_actions:
                            if 0 <= action < 9:
                                self.action_counts[action] += 1
                        
                        # Calculate entropy of action distribution
                        total_actions = sum(self.action_counts.values())
                        if total_actions > 0:
                            action_probs = [count / total_actions for count in self.action_counts.values()]
                            entropy = -sum(p * np.log(p + 1e-8) for p in action_probs if p > 0)
                            
                            # If entropy is too low, increase entropy coefficient
                            if entropy < self.min_entropy_threshold:
                                current_ent_coef = max(current_ent_coef, 0.2)  # Emergency entropy boost
                                self.logger.record('entropy_scheduler/emergency_boost', 1.0)
                            
                            self.logger.record('entropy_scheduler/action_entropy', entropy)
                            
                        # Reset counts periodically
                        if self.num_timesteps % 10000 == 0:
                            self.action_counts = {i: 0 for i in range(9)}
                            
                except Exception as e:
                    # Fallback if monitoring fails
                    pass
            
            # Update model entropy coefficient
            self.model.ent_coef = current_ent_coef
            
            # Log every 1000 steps
            if self.num_timesteps % 1000 == 0:
                self.logger.record('entropy_scheduler/ent_coef', current_ent_coef)
                self.logger.record('entropy_scheduler/progress', progress)
                
            return True
    
    return EnhancedEntropyScheduler