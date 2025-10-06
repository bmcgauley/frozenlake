"""
Screen State Tracking Utility for Pokemon Red RL Agent

This module provides utilities to track and analyze screen states to detect
when the agent is stuck on repeated screens (like title screens or menus).

Features:
- Efficient screen hashing using perceptual hashing
- History-based duplicate detection
- Escalating penalties for revisiting states
- Progress detection for key game milestones
"""

import numpy as np
from collections import deque, Counter
from typing import Optional, Tuple, Dict
import hashlib


class ScreenStateTracker:
    """
    Tracks screen states to detect stagnation and repeated states.
    
    This class maintains a history of screen hashes and provides methods
    to detect when the agent is stuck in loops or not making progress.
    """
    
    def __init__(self, 
                 history_size: int = 100,
                 short_term_size: int = 20,
                 pixel_similarity_threshold: float = 0.95):
        """
        Initialize the screen state tracker.
        
        Args:
            history_size: Number of screen states to remember
            short_term_size: Size of short-term memory for immediate loop detection
            pixel_similarity_threshold: Threshold for considering screens identical (0-1)
        """
        self.history_size = history_size
        self.short_term_size = short_term_size
        self.pixel_similarity_threshold = pixel_similarity_threshold
        
        # Long-term screen history (last N screen hashes)
        self.screen_history = deque(maxlen=history_size)
        
        # Short-term screen history (for immediate loop detection)
        self.short_term_history = deque(maxlen=short_term_size)
        
        # Counter for frequency of each screen hash
        self.screen_frequencies = Counter()
        
        # Track consecutive steps on same screen
        self.consecutive_same_screen = 0
        self.last_screen_hash = None
        
        # Track known "progress" screens we've passed
        self.progress_screens = set()
        
        # Statistics
        self.total_steps = 0
        self.unique_screens_seen = 0
        self.loop_detections = 0
    
    def hash_screen(self, screen: np.ndarray) -> str:
        """
        Create a hash of the screen state.
        
        Uses MD5 hash of screen pixels for fast, reliable comparison.
        
        Args:
            screen: Screen array (H, W, 3) RGB
            
        Returns:
            String hash of the screen
        """
        # Convert to bytes and hash
        screen_bytes = screen.tobytes()
        return hashlib.md5(screen_bytes).hexdigest()
    
    def hash_screen_downsampled(self, screen: np.ndarray, size: Tuple[int, int] = (32, 32)) -> str:
        """
        Create a perceptual hash of the screen by downsampling.
        
        This is more tolerant to small changes (like blinking cursors or animations)
        but still detects major screen differences.
        
        Args:
            screen: Screen array (H, W, 3) RGB
            size: Size to downsample to for hashing
            
        Returns:
            String hash of the downsampled screen
        """
        from skimage.transform import resize
        
        # Downsample to small size
        downsampled = resize(
            screen, 
            size, 
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.uint8)
        
        # Hash the downsampled version
        return hashlib.md5(downsampled.tobytes()).hexdigest()
    
    def update(self, screen: np.ndarray) -> Dict[str, any]:
        """
        Update tracker with new screen state.
        
        Args:
            screen: Current screen array (H, W, 3) RGB
            
        Returns:
            Dictionary with analysis results:
                - 'is_duplicate': Is this screen in recent history?
                - 'duplicate_count': How many times this screen appears in history
                - 'recency': How recently was this screen seen? (0=just seen, 1=oldest)
                - 'is_stuck': Is the agent stuck on the same screen?
                - 'consecutive_same': Number of consecutive steps on same screen
                - 'is_loop': Is the agent in a short loop?
                - 'loop_length': If looping, length of the loop
        """
        self.total_steps += 1
        
        # Hash the screen (use downsampled for slight tolerance)
        screen_hash = self.hash_screen_downsampled(screen)
        
        # Check if this is a new unique screen
        if screen_hash not in self.screen_frequencies:
            self.unique_screens_seen += 1
        
        # Update frequency counter
        self.screen_frequencies[screen_hash] += 1
        
        # Check for duplicate in recent history
        is_duplicate = screen_hash in self.screen_history
        duplicate_count = self.screen_frequencies[screen_hash]
        
        # Calculate recency (how recently was this seen?)
        recency = 1.0  # Default: not recently seen
        if is_duplicate:
            try:
                # Find most recent occurrence
                history_list = list(self.screen_history)
                last_index = len(history_list) - 1 - history_list[::-1].index(screen_hash)
                # Recency: 0 = just seen, 1 = oldest in history
                recency = (len(history_list) - last_index - 1) / len(history_list)
            except ValueError:
                pass
        
        # Check for consecutive same screen (stuck)
        is_stuck = (screen_hash == self.last_screen_hash)
        if is_stuck:
            self.consecutive_same_screen += 1
        else:
            self.consecutive_same_screen = 0
        
        # Check for short loops (e.g., alternating between 2-3 screens)
        is_loop = False
        loop_length = 0
        if len(self.short_term_history) >= 4:
            # Check for repeating patterns
            recent = list(self.short_term_history)[-10:]
            
            # Try to detect loops of length 2-5
            for loop_len in range(2, 6):
                if len(recent) >= loop_len * 2:
                    # Check if pattern repeats
                    pattern = recent[-loop_len:]
                    previous = recent[-loop_len*2:-loop_len]
                    
                    if pattern == previous:
                        is_loop = True
                        loop_length = loop_len
                        self.loop_detections += 1
                        break
        
        # Update histories
        self.screen_history.append(screen_hash)
        self.short_term_history.append(screen_hash)
        self.last_screen_hash = screen_hash
        
        return {
            'screen_hash': screen_hash,
            'is_duplicate': is_duplicate,
            'duplicate_count': duplicate_count,
            'recency': recency,
            'is_stuck': is_stuck,
            'consecutive_same': self.consecutive_same_screen,
            'is_loop': is_loop,
            'loop_length': loop_length,
            'unique_screens': self.unique_screens_seen,
            'total_steps': self.total_steps,
        }
    
    def calculate_stagnation_penalty(self, analysis: Dict[str, any]) -> float:
        """
        Calculate a penalty based on screen state analysis.
        
        Penalties are applied for:
        - Being stuck on the same screen (consecutive same)
        - Revisiting recently seen screens
        - Being in a short loop
        - High frequency screens (seen many times)
        
        Args:
            analysis: Output from update() method
            
        Returns:
            Penalty value (negative float)
        """
        penalty = 0.0
        
        # 1. STUCK PENALTY - Consecutive same screen
        if analysis['is_stuck']:
            consecutive = analysis['consecutive_same']
            # REDUCED escalating penalty: -0.2, -0.3, -0.4, -0.5, ...
            # Combined with high entropy (exploration), this should be enough
            penalty -= 0.2 * (1 + consecutive / 20.0)
        
        # 2. DUPLICATE PENALTY - Seen this screen before
        if analysis['is_duplicate']:
            # Penalty based on how recently it was seen
            # Recent duplicates are worse (suggests tight loop)
            recency = analysis['recency']
            
            # REDUCED penalties - let exploration drive the agent
            # If seen very recently (recency < 0.2), harsh penalty
            if recency < 0.2:
                penalty -= 0.5  # Reduced from 2.0
            # If seen somewhat recently (recency < 0.5), moderate penalty
            elif recency < 0.5:
                penalty -= 0.2  # Reduced from 1.0
            # If seen long ago (recency >= 0.5), tiny penalty
            else:
                penalty -= 0.05  # Reduced from 0.3
            
            # Additional penalty for high frequency screens
            duplicate_count = analysis['duplicate_count']
            if duplicate_count > 10:
                penalty -= 0.1 * (duplicate_count / 10.0)  # Reduced from 0.5
        
        # 3. LOOP PENALTY - Stuck in a short repeating loop
        if analysis['is_loop']:
            loop_length = analysis['loop_length']
            # Shorter loops are worse (more stuck)
            penalty -= 1.0 / loop_length  # Reduced from 5.0
        
        return penalty
    
    def calculate_progress_reward(self, analysis: Dict[str, any]) -> float:
        """
        Calculate a reward for making progress (seeing new screens).
        
        Args:
            analysis: Output from update() method
            
        Returns:
            Reward value (positive float)
        """
        reward = 0.0
        
        # NEW SCREEN REWARD
        if not analysis['is_duplicate']:
            # Seeing a brand new screen is GOOD!
            reward += 1.0
            
            # Mark as progress screen
            self.progress_screens.add(analysis['screen_hash'])
        
        # PROGRESS REWARD - Moving forward (not stuck)
        if not analysis['is_stuck']:
            # Small reward for any change
            reward += 0.1
        
        return reward
    
    def detect_title_screen(self, screen: np.ndarray) -> bool:
        """
        Detect if current screen is the Pokemon Red title screen.
        
        This uses pattern matching on the screen content.
        The title screen has distinctive features we can detect.
        
        Args:
            screen: Screen array (H, W, 3) RGB
            
        Returns:
            True if this appears to be the title screen
        """
        # Simple heuristic: Title screen has lots of red/yellow colors
        # and specific patterns
        
        # Check if screen has high red content (Pokémon logo is red)
        red_channel = screen[:, :, 0]
        mean_red = np.mean(red_channel)
        
        # Check for low variation (title screen is mostly static text/logo)
        std_red = np.std(red_channel)
        
        # Title screen typically has:
        # - High mean red value (logo)
        # - Moderate std (not all one color, but not chaotic)
        # This is a rough heuristic - adjust thresholds as needed
        is_title = (mean_red > 100 and std_red > 30 and std_red < 80)
        
        return is_title
    
    def detect_menu_screen(self, screen: np.ndarray) -> bool:
        """
        Detect if current screen is a menu (pause menu, start menu, etc).
        
        Menus typically have:
        - Text boxes with black/white borders
        - Less varied colors than gameplay
        
        Args:
            screen: Screen array (H, W, 3) RGB
            
        Returns:
            True if this appears to be a menu screen
        """
        # Calculate color variance
        color_std = np.std(screen, axis=(0, 1))
        
        # Menus have lower color variance
        mean_std = np.mean(color_std)
        
        # Also check for predominance of white/black (text boxes)
        grayscale = np.mean(screen, axis=2)
        white_pixels = np.sum(grayscale > 200)
        black_pixels = np.sum(grayscale < 50)
        total_pixels = grayscale.size
        
        text_box_ratio = (white_pixels + black_pixels) / total_pixels
        
        # Menu detection heuristic
        is_menu = (mean_std < 40 and text_box_ratio > 0.3)
        
        return is_menu
    
    def get_statistics(self) -> Dict[str, any]:
        """Get tracking statistics."""
        return {
            'total_steps': self.total_steps,
            'unique_screens': self.unique_screens_seen,
            'loop_detections': self.loop_detections,
            'diversity_ratio': self.unique_screens_seen / max(1, self.total_steps),
            'most_common_screens': self.screen_frequencies.most_common(5),
        }
    
    def reset(self):
        """Reset tracker for new episode."""
        self.consecutive_same_screen = 0
        self.last_screen_hash = None
        # Don't reset screen_frequencies - we want to remember across episodes
        # But do reset short-term history
        self.short_term_history.clear()
