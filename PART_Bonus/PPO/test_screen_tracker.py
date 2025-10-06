"""
Test Script for Screen State Tracker

This script demonstrates how the screen state tracker works
without requiring a trained model. It simulates the agent
stuck on the title screen to show how penalties escalate.
"""

import numpy as np
from screen_state_tracker import ScreenStateTracker


def create_fake_screen(pattern='title'):
    """Create a fake screen for testing."""
    screen = np.zeros((144, 160, 3), dtype=np.uint8)
    
    if pattern == 'title':
        # Simulate title screen with red/yellow
        screen[:, :, 0] = 150  # Red channel
        screen[60:80, 40:120, 0] = 200  # Brighter red block (logo)
        screen[60:80, 40:120, 1] = 50   # Yellow tint
        
    elif pattern == 'menu':
        # Simulate menu with black/white
        screen[:, :] = 200  # White background
        screen[20:40, 20:140, :] = 0  # Black text box
        screen[50:70, 20:140, :] = 0  # Black text box
        
    elif pattern == 'gameplay':
        # Simulate gameplay with varied colors
        screen[:, :, 1] = 100  # Green grass
        screen[50:60, 70:80, :] = [200, 0, 0]  # Red sprite (player)
        screen[70:100, 40:120, 2] = 150  # Blue water
        
    elif pattern == 'gameplay2':
        # Similar to gameplay but player moved
        screen[:, :, 1] = 100  # Green grass
        screen[52:62, 72:82, :] = [200, 0, 0]  # Red sprite (player moved)
        screen[70:100, 40:120, 2] = 150  # Blue water
        
    return screen


def test_screen_tracker():
    """Test the screen state tracker with various scenarios."""
    
    print("=" * 80)
    print("SCREEN STATE TRACKER TEST")
    print("=" * 80)
    
    tracker = ScreenStateTracker(history_size=50, short_term_size=10)
    
    print("\n" + "=" * 80)
    print("SCENARIO 1: Stuck on Title Screen")
    print("=" * 80)
    
    title_screen = create_fake_screen('title')
    
    total_reward = 0.0
    for step in range(20):
        analysis = tracker.update(title_screen)
        penalty = tracker.calculate_stagnation_penalty(analysis)
        reward = tracker.calculate_progress_reward(analysis)
        net_reward = penalty + reward
        total_reward += net_reward
        
        status = []
        if analysis['is_stuck']:
            status.append(f"STUCK({analysis['consecutive_same']})")
        if analysis['is_duplicate']:
            status.append(f"DUP({analysis['duplicate_count']})")
        if analysis['is_loop']:
            status.append(f"LOOP({analysis['loop_length']})")
        
        is_title = tracker.detect_title_screen(title_screen)
        if is_title:
            status.append("TITLE")
        
        status_str = " | ".join(status) if status else "NEW"
        
        print(f"Step {step:2d}: Penalty={penalty:+6.2f}, Reward={reward:+5.2f}, "
              f"Net={net_reward:+6.2f}, Total={total_reward:+7.2f} | {status_str}")
    
    print(f"\nResult: After 20 steps on title screen, total reward = {total_reward:+.2f}")
    print("Notice how penalties escalate rapidly!")
    
    print("\n" + "=" * 80)
    print("SCENARIO 2: Progressing Through Different Screens")
    print("=" * 80)
    
    # Reset for new scenario
    tracker = ScreenStateTracker(history_size=50, short_term_size=10)
    
    screens = [
        create_fake_screen('title'),
        create_fake_screen('menu'),
        create_fake_screen('gameplay'),
        create_fake_screen('gameplay2'),
        create_fake_screen('gameplay'),  # Revisit
    ]
    
    screen_names = ['Title', 'Menu', 'Gameplay', 'Gameplay2', 'Gameplay(revisit)']
    
    total_reward = 0.0
    for step, (screen, name) in enumerate(zip(screens * 4, screen_names * 4)):
        analysis = tracker.update(screen)
        penalty = tracker.calculate_stagnation_penalty(analysis)
        reward = tracker.calculate_progress_reward(analysis)
        net_reward = penalty + reward
        total_reward += net_reward
        
        status = []
        if analysis['is_stuck']:
            status.append(f"STUCK({analysis['consecutive_same']})")
        if analysis['is_duplicate']:
            status.append(f"DUP({analysis['duplicate_count']})")
        if analysis['is_loop']:
            status.append(f"LOOP({analysis['loop_length']})")
        if not status:
            status.append("NEW" if not analysis['is_duplicate'] else "SEEN")
        
        status_str = " | ".join(status)
        
        print(f"Step {step:2d} ({name:20s}): Penalty={penalty:+6.2f}, Reward={reward:+5.2f}, "
              f"Net={net_reward:+6.2f}, Total={total_reward:+7.2f} | {status_str}")
        
        if step >= 19:  # Only show first 20 steps
            break
    
    print(f"\nResult: Total reward = {total_reward:+.2f}")
    print("Notice how new screens are rewarded, but loops are penalized!")
    
    print("\n" + "=" * 80)
    print("SCENARIO 3: Detecting a 2-Screen Loop")
    print("=" * 80)
    
    # Reset for new scenario
    tracker = ScreenStateTracker(history_size=50, short_term_size=10)
    
    screen_a = create_fake_screen('menu')
    screen_b = create_fake_screen('gameplay')
    
    total_reward = 0.0
    for step in range(20):
        # Alternate between two screens (simulating opening/closing menu)
        screen = screen_a if step % 2 == 0 else screen_b
        screen_name = "Menu" if step % 2 == 0 else "Gameplay"
        
        analysis = tracker.update(screen)
        penalty = tracker.calculate_stagnation_penalty(analysis)
        reward = tracker.calculate_progress_reward(analysis)
        net_reward = penalty + reward
        total_reward += net_reward
        
        status = []
        if analysis['is_stuck']:
            status.append(f"STUCK({analysis['consecutive_same']})")
        if analysis['is_duplicate']:
            status.append(f"DUP({analysis['duplicate_count']})")
        if analysis['is_loop']:
            status.append(f"LOOP({analysis['loop_length']})")
        
        status_str = " | ".join(status) if status else "NEW"
        
        print(f"Step {step:2d} ({screen_name:10s}): Penalty={penalty:+6.2f}, Reward={reward:+5.2f}, "
              f"Net={net_reward:+6.2f}, Total={total_reward:+7.2f} | {status_str}")
    
    print(f"\nResult: After 20 steps alternating between 2 screens, total reward = {total_reward:+.2f}")
    print("Notice how loop detection kicks in after a few cycles!")
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("TRACKER STATISTICS")
    print("=" * 80)
    stats = tracker.get_statistics()
    print(f"Total steps: {stats['total_steps']}")
    print(f"Unique screens: {stats['unique_screens']}")
    print(f"Diversity ratio: {stats['diversity_ratio']:.2%}")
    print(f"Loop detections: {stats['loop_detections']}")
    print(f"\nMost common screens:")
    for screen_hash, count in stats['most_common_screens']:
        print(f"  {screen_hash[:16]}... appeared {count} times")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Being stuck on the same screen results in escalating penalties")
    print("2. Revisiting recently seen screens is penalized")
    print("3. Short loops (alternating between 2-3 screens) are detected and penalized")
    print("4. Seeing new screens is rewarded")
    print("5. Title screens and menus can be detected visually")


if __name__ == '__main__':
    test_screen_tracker()
