#!/usr/bin/env python3
"""
CNN Debug Analyzer - Visualize what the Pokemon Red RL model sees

This script helps analyze the CNN input debug data to understand:
1. What the 3-frame stacked input looks like to the model
2. How the status bars overlay affects the input 
3. Temporal changes in the model's perception
4. Action-observation relationships

Usage:
    python cnn_debug_analyzer.py [session_path] [options]

Features:
- Create videos showing model perception over time
- Analyze frame differences and movement patterns
- Generate summary reports of exploration patterns
- Compare different training sessions
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json
from datetime import datetime

def find_latest_session():
    """Find the most recent training session directory."""
    sessions_dir = Path('./sessions')
    if not sessions_dir.exists():
        return None
    
    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir() and d.name.startswith('pokemon_rl_')]
    if not session_dirs:
        return None
    
    # Sort by timestamp in directory name
    session_dirs.sort(key=lambda x: x.name)
    return session_dirs[-1]

def analyze_cnn_debug_data(session_path, max_frames=500):
    """
    Analyze CNN debug data from a training session.
    
    Args:
        session_path: Path to session directory
        max_frames: Maximum number of frames to analyze
    """
    session_path = Path(session_path)
    
    # Look for CNN debug directories
    cnn_debug_dirs = []
    for item in session_path.iterdir():
        if item.is_dir() and 'cnn_debug' in item.name:
            cnn_debug_dirs.append(item)
    
    if not cnn_debug_dirs:
        print(f"❌ No CNN debug directories found in {session_path}")
        print("Make sure debug_cnn_input=True was set during training")
        return
    
    print(f"🔬 Found {len(cnn_debug_dirs)} CNN debug directories")
    
    for debug_dir in cnn_debug_dirs:
        print(f"\n📁 Analyzing: {debug_dir.name}")
        
        # Find all debug images
        debug_images = sorted(list(debug_dir.glob("cnn_input_*.png")))
        debug_data = sorted(list(debug_dir.glob("cnn_data_*.npy")))
        
        print(f"   🖼️  Images found: {len(debug_images)}")
        print(f"   📊 Data files found: {len(debug_data)}")
        
        if not debug_images:
            print("   ⚠️  No debug images found")
            continue
        
        # Limit analysis to max_frames
        if len(debug_images) > max_frames:
            debug_images = debug_images[-max_frames:]
            debug_data = debug_data[-max_frames:]
            print(f"   📉 Limited analysis to most recent {max_frames} frames")
        
        # Analyze temporal patterns
        analyze_temporal_patterns(debug_dir, debug_data[:min(100, len(debug_data))])
        
        # Create perception video
        create_perception_video(debug_dir, debug_images)
        
        # Generate analysis report
        generate_analysis_report(debug_dir, debug_data)

def analyze_temporal_patterns(debug_dir, data_files):
    """Analyze how the model's perception changes over time."""
    print("   🕐 Analyzing temporal patterns...")
    
    if len(data_files) < 10:
        print("   ⚠️  Not enough data for temporal analysis")
        return
    
    frame_differences = []
    movement_patterns = []
    
    prev_observation = None
    
    for data_file in data_files[:50]:  # Analyze first 50 for speed
        try:
            observation = np.load(data_file)
            
            if prev_observation is not None:
                # Calculate frame difference
                diff = np.mean(np.abs(observation - prev_observation))
                frame_differences.append(diff)
                
                # Detect movement by comparing most recent frames
                current_frame = observation[:, :, -1]  # Most recent frame
                prev_frame = prev_observation[:, :, -1]
                movement = np.mean(np.abs(current_frame - prev_frame))
                movement_patterns.append(movement)
            
            prev_observation = observation
            
        except Exception as e:
            print(f"   ⚠️  Error loading {data_file}: {e}")
            continue
    
    if frame_differences:
        # Create temporal analysis plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(frame_differences)
        ax1.set_title('Frame Differences Over Time (Model Perception Changes)')
        ax1.set_ylabel('Average Pixel Difference')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(movement_patterns, color='orange')
        ax2.set_title('Movement Detection (Screen Motion)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Movement Intensity')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save analysis plot
        analysis_plot_path = debug_dir / 'temporal_analysis.png'
        plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Temporal analysis saved to: {analysis_plot_path}")
        
        # Calculate statistics
        avg_change = np.mean(frame_differences)
        max_change = np.max(frame_differences)
        avg_movement = np.mean(movement_patterns)
        
        print(f"   📊 Average perception change: {avg_change:.2f}")
        print(f"   📊 Maximum perception change: {max_change:.2f}")
        print(f"   📊 Average movement detected: {avg_movement:.2f}")

def create_perception_video(debug_dir, image_files, fps=8):
    """Create a video showing the model's perception over time."""
    print("   🎬 Creating perception video...")
    
    try:
        import cv2
        
        if len(image_files) < 5:
            print("   ⚠️  Not enough images for video creation")
            return
        
        # Read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            print("   ❌ Failed to read first image")
            return
        
        height, width, layers = first_img.shape
        
        # Create video writer
        video_path = debug_dir / "model_perception_timeline.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        frames_processed = 0
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                video_writer.write(img)
                frames_processed += 1
        
        video_writer.release()
        
        print(f"   🎥 Video created: {video_path}")
        print(f"   🎥 Processed {frames_processed} frames at {fps} FPS")
        print(f"   🎥 Video duration: {frames_processed/fps:.1f} seconds")
        
    except ImportError:
        print("   ❌ OpenCV not available. Install with: pip install opencv-python")
    except Exception as e:
        print(f"   ❌ Video creation failed: {e}")

def generate_analysis_report(debug_dir, data_files):
    """Generate a comprehensive analysis report."""
    print("   📝 Generating analysis report...")
    
    if not data_files:
        return
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'session_path': str(debug_dir.parent),
        'total_frames_analyzed': len(data_files),
        'frame_statistics': {},
        'observations': []
    }
    
    # Analyze sample frames
    sample_indices = np.linspace(0, len(data_files)-1, min(20, len(data_files)), dtype=int)
    
    intensity_values = []
    
    for i, idx in enumerate(sample_indices):
        try:
            observation = np.load(data_files[idx])
            
            # Calculate frame statistics
            mean_intensity = np.mean(observation)
            std_intensity = np.std(observation)
            min_intensity = np.min(observation)
            max_intensity = np.max(observation)
            
            intensity_values.append(mean_intensity)
            
            frame_stats = {
                'frame_index': int(idx),
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'min_intensity': int(min_intensity),
                'max_intensity': int(max_intensity),
                'unique_values': int(len(np.unique(observation)))
            }
            
            if i < 5:  # Detailed analysis for first few frames
                report['observations'].append(frame_stats)
        
        except Exception as e:
            print(f"   ⚠️  Error analyzing frame {idx}: {e}")
    
    # Overall statistics
    if intensity_values:
        report['frame_statistics'] = {
            'mean_intensity_across_time': float(np.mean(intensity_values)),
            'intensity_variation': float(np.std(intensity_values)),
            'intensity_range': [float(np.min(intensity_values)), float(np.max(intensity_values))]
        }
    
    # Save report
    report_path = debug_dir / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   📋 Analysis report saved to: {report_path}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Analyze CNN debug data from Pokemon Red RL training')
    parser.add_argument('session_path', nargs='?', help='Path to session directory (auto-detects latest if not provided)')
    parser.add_argument('--max-frames', type=int, default=500, help='Maximum frames to analyze (default: 500)')
    parser.add_argument('--video-fps', type=int, default=8, help='FPS for generated videos (default: 8)')
    
    args = parser.parse_args()
    
    # Determine session path
    if args.session_path:
        session_path = Path(args.session_path)
    else:
        session_path = find_latest_session()
    
    if not session_path or not session_path.exists():
        print("❌ No valid session directory found")
        print("Usage:")
        print("  python cnn_debug_analyzer.py [session_path]")
        print("  python cnn_debug_analyzer.py  # Auto-detect latest session")
        return 1
    
    print(f"🔬 CNN Debug Analyzer")
    print(f"📁 Session: {session_path}")
    print(f"⚙️  Max frames: {args.max_frames}")
    print("=" * 60)
    
    # Run analysis
    analyze_cnn_debug_data(session_path, args.max_frames)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Check {session_path} for generated analysis files:")
    print("   - model_perception_timeline.mp4 (video)")
    print("   - temporal_analysis.png (charts)")
    print("   - analysis_report.json (detailed stats)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())