# FIXED REWARD SYSTEM - Prevents Policy Collapse

def _calculate_reward_fixed(self):
    """
    IMPROVED REWARD STRUCTURE - Prevents policy collapse while encouraging exploration
    
    Key Changes:
    1. Balanced reward scales - no single reward dominates
    2. Action diversity bonus - prevents spamming single action
    3. Movement incentives - encourages spatial exploration
    4. Reasonable penalties - discourages bad behavior without fear
    """
    total_reward = 0.0
    
    # ====================================================================
    # 1. BASE TIME PENALTY - Encourage action taking
    # ====================================================================
    time_penalty = -0.005  # Reduced from -0.01 to be less punishing
    total_reward += time_penalty
    
    # ====================================================================
    # 2. VISUAL NOVELTY REWARD - Balanced exploration incentive
    # ====================================================================
    screen_image = self.pyboy.screen.image.convert('RGB')
    screen_array = np.asarray(screen_image)
    screen_analysis = self.screen_tracker.update(screen_array)
    
    # Reward new screens but not overpowering
    if screen_analysis.get('is_new_screen', False):
        novelty_reward = 2.0  # Reduced from potentially huge rewards
        total_reward += novelty_reward
    
    # Small diversity bonus
    diversity_bonus = screen_analysis.get('diversity_score', 0) * 0.2
    total_reward += diversity_bonus
    
    # ====================================================================
    # 3. ACTION DIVERSITY BONUS - Prevents action spamming
    # ====================================================================
    if len(self.action_history) >= 10:
        # Calculate action entropy over last 10 actions
        recent_actions = list(self.action_history)[-10:]
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate entropy (diversity)
        total_actions = len(recent_actions)
        entropy = 0
        for count in action_counts.values():
            prob = count / total_actions
            if prob > 0:
                entropy -= prob * np.log(prob)
        
        # Normalize entropy (max entropy for 9 actions is ~2.2)
        max_entropy = np.log(9)  # ~2.2
        diversity_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        # Reward action diversity
        diversity_bonus = diversity_ratio * 0.5  # Up to 0.5 bonus for full diversity
        total_reward += diversity_bonus
        
        # Penalty for excessive repetition (>80% same action)
        most_common_count = max(action_counts.values())
        if most_common_count > 8:  # More than 8/10 same action
            repetition_penalty = -1.0
            total_reward += repetition_penalty
    
    # ====================================================================
    # 4. MOVEMENT REWARD - Encourage spatial exploration
    # ====================================================================
    current_pos = (
        self._read_memory(self.PLAYER_X_ADDRESS),
        self._read_memory(self.PLAYER_Y_ADDRESS),
        self._read_memory(self.MAP_ID_ADDRESS)
    )
    
    # Track visited positions
    if not hasattr(self, 'visited_positions'):
        self.visited_positions = set()
    
    if current_pos not in self.visited_positions:
        self.visited_positions.add(current_pos)
        movement_reward = 1.0  # Meaningful reward for new locations
        total_reward += movement_reward
    
    # ====================================================================
    # 5. ANTI-STUCK PENALTIES - Reasonable deterrents
    # ====================================================================
    
    # Penalty for being stuck in same screen too long
    if screen_analysis.get('consecutive_same', 0) > 20:
        stuck_penalty = -0.1  # Gentle penalty, not overwhelming
        total_reward += stuck_penalty
    
    # Small penalty for no-op action to encourage engagement
    if self.last_action == 8:  # 8 = no-op/wait
        idle_penalty = -0.02
        total_reward += idle_penalty
    
    return total_reward