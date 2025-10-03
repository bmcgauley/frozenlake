"""
Pokemon Red RL - Swarm Learning Coordinator

This script coordinates multiple training agents to share knowledge:
- Agents train independently in parallel
- Periodically share best model weights
- Aggregate exploration data across all agents
- Synchronize progress for faster convergence

Architecture:
- Central coordinator manages weight sharing
- Each agent trains with local rewards
- Best-performing agent's weights are shared
- Agents can specialize in different game aspects

Usage:
    # Start coordinator (in one terminal)
    python swarm_coordinator.py --agents 4 --sync-freq 100000

    # Agents automatically managed by coordinator
    # Or run agents manually on different machines and specify coordinator IP
"""

import os
import time
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import threading
import queue

print("=" * 80)
print("POKEMON RED RL - SWARM LEARNING COORDINATOR")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser(description='Swarm Learning Coordinator')
parser.add_argument('--agents', type=int, default=4,
                   help='Number of parallel agents')
parser.add_argument('--sync-freq', type=int, default=100000,
                   help='Synchronization frequency (timesteps)')
parser.add_argument('--strategy', choices=['best', 'ensemble', 'diverse'], 
                   default='best',
                   help='Weight sharing strategy')
parser.add_argument('--session-dir', type=str, default='./swarm_sessions',
                   help='Directory for swarm sessions')
args = parser.parse_args()

# ============================================================================
# SWARM COORDINATOR
# ============================================================================

class SwarmCoordinator:
    """
    Coordinates multiple RL agents for collaborative learning.
    
    Each agent trains independently but shares knowledge periodically.
    The coordinator tracks performance and distributes best practices.
    """
    
    def __init__(self, num_agents, sync_frequency, strategy='best', session_dir='./swarm_sessions'):
        self.num_agents = num_agents
        self.sync_frequency = sync_frequency
        self.strategy = strategy
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        
        # Create session directory
        self.swarm_id = f"swarm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.swarm_path = self.session_dir / self.swarm_id
        self.swarm_path.mkdir(exist_ok=True)
        
        # Shared directories
        self.weights_dir = self.swarm_path / 'shared_weights'
        self.weights_dir.mkdir(exist_ok=True)
        
        self.metrics_dir = self.swarm_path / 'metrics'
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.exploration_dir = self.swarm_path / 'exploration'
        self.exploration_dir.mkdir(exist_ok=True)
        
        # Agent tracking
        self.agents = {}
        self.agent_stats = {}
        self.best_agent = None
        self.best_reward = -float('inf')
        
        # Synchronization tracking
        self.last_sync = 0
        self.sync_count = 0
        
        # Exploration aggregation
        self.global_exploration = set()
        
        print(f"\n🐝 Swarm Coordinator Initialized")
        print(f"   Swarm ID: {self.swarm_id}")
        print(f"   Agents: {num_agents}")
        print(f"   Sync Frequency: {sync_frequency:,} steps")
        print(f"   Strategy: {strategy}")
        print(f"   Session: {self.swarm_path}")
    
    def register_agent(self, agent_id):
        """Register a new agent with the swarm."""
        self.agents[agent_id] = {
            'status': 'active',
            'timesteps': 0,
            'reward': 0,
            'badges': 0,
            'exploration': 0,
            'last_update': time.time()
        }
        
        self.agent_stats[agent_id] = {
            'rewards': [],
            'badges': [],
            'timesteps': [],
            'exploration': []
        }
        
        print(f"\n✅ Agent {agent_id} registered")
    
    def update_agent_metrics(self, agent_id, metrics):
        """Update metrics for a specific agent."""
        if agent_id not in self.agents:
            self.register_agent(agent_id)
        
        # Update current state
        self.agents[agent_id].update({
            'timesteps': metrics.get('timesteps', 0),
            'reward': metrics.get('reward', 0),
            'badges': metrics.get('badges', 0),
            'exploration': metrics.get('exploration', 0),
            'last_update': time.time()
        })
        
        # Track history
        self.agent_stats[agent_id]['timesteps'].append(metrics.get('timesteps', 0))
        self.agent_stats[agent_id]['rewards'].append(metrics.get('reward', 0))
        self.agent_stats[agent_id]['badges'].append(metrics.get('badges', 0))
        self.agent_stats[agent_id]['exploration'].append(metrics.get('exploration', 0))
        
        # Check if this is best agent
        if metrics.get('reward', 0) > self.best_reward:
            self.best_reward = metrics.get('reward', 0)
            self.best_agent = agent_id
            print(f"\n⭐ New best agent: Agent {agent_id} (reward: {self.best_reward:.2f})")
    
    def should_sync(self):
        """Check if it's time to synchronize."""
        max_timesteps = max(agent['timesteps'] for agent in self.agents.values())
        
        if max_timesteps - self.last_sync >= self.sync_frequency:
            return True
        return False
    
    def synchronize_weights(self):
        """
        Synchronize model weights across agents based on strategy.
        
        Strategies:
        - 'best': Share weights from best-performing agent
        - 'ensemble': Average weights from all agents
        - 'diverse': Share different weights to maintain diversity
        """
        print(f"\n🔄 Synchronization #{self.sync_count + 1}")
        print("=" * 60)
        
        if self.strategy == 'best':
            # Share best agent's weights
            if self.best_agent is not None:
                print(f"Strategy: Sharing weights from Agent {self.best_agent}")
                print(f"Best Reward: {self.best_reward:.2f}")
                
                # Copy best model to shared directory
                best_model_src = self.swarm_path / f"agent_{self.best_agent}" / "current_model.zip"
                best_model_dst = self.weights_dir / f"best_model_sync{self.sync_count}.zip"
                
                if best_model_src.exists():
                    shutil.copy2(best_model_src, best_model_dst)
                    
                    # Notify all agents to load new weights
                    sync_info = {
                        'sync_id': self.sync_count,
                        'best_agent': self.best_agent,
                        'best_reward': self.best_reward,
                        'model_path': str(best_model_dst),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    sync_file = self.weights_dir / 'latest_sync.json'
                    with open(sync_file, 'w') as f:
                        json.dump(sync_info, f, indent=2)
                    
                    print(f"✅ Weights saved: {best_model_dst}")
        
        elif self.strategy == 'ensemble':
            print("Strategy: Averaging weights from all agents")
            # Would require loading all models and averaging parameters
            # Left as placeholder for advanced implementation
            print("⚠️  Ensemble averaging not yet implemented")
        
        elif self.strategy == 'diverse':
            print("Strategy: Maintaining diversity across agents")
            # Each agent keeps its own weights but receives others' exploration data
            print("✅ Agents maintain independent weights")
        
        # Update sync tracking
        self.last_sync = max(agent['timesteps'] for agent in self.agents.values())
        self.sync_count += 1
        
        # Save swarm statistics
        self.save_swarm_stats()
        
        print("=" * 60)
    
    def aggregate_exploration(self):
        """Aggregate exploration data from all agents."""
        # Load exploration data from all agents
        for agent_id in self.agents:
            exploration_file = self.exploration_dir / f"agent_{agent_id}_coords.json"
            
            if exploration_file.exists():
                try:
                    with open(exploration_file, 'r') as f:
                        agent_coords = json.load(f)
                        # Convert list of tuples back to set
                        self.global_exploration.update([tuple(c) for c in agent_coords])
                except Exception as e:
                    print(f"⚠️  Could not load exploration for agent {agent_id}: {e}")
        
        # Save global exploration map
        global_coords = list(self.global_exploration)
        global_file = self.exploration_dir / 'global_exploration.json'
        with open(global_file, 'w') as f:
            json.dump(global_coords, f)
        
        print(f"\n🗺️  Global Exploration: {len(self.global_exploration)} unique coordinates")
    
    def save_swarm_stats(self):
        """Save swarm-wide statistics."""
        stats = {
            'swarm_id': self.swarm_id,
            'num_agents': self.num_agents,
            'sync_count': self.sync_count,
            'best_agent': self.best_agent,
            'best_reward': self.best_reward,
            'global_exploration': len(self.global_exploration),
            'agents': {
                agent_id: {
                    'current': agent_data,
                    'history': self.agent_stats[agent_id]
                }
                for agent_id, agent_data in self.agents.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        stats_file = self.metrics_dir / f'swarm_stats_sync{self.sync_count}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Statistics saved: {stats_file}")
    
    def print_status(self):
        """Print current swarm status."""
        print("\n" + "=" * 80)
        print(f"SWARM STATUS - Sync #{self.sync_count}")
        print("=" * 80)
        
        # Sort agents by reward
        sorted_agents = sorted(
            self.agents.items(),
            key=lambda x: x[1]['reward'],
            reverse=True
        )
        
        print(f"\n{'Agent':<10} {'Timesteps':<12} {'Reward':<12} {'Badges':<8} {'Explored':<10}")
        print("-" * 80)
        
        for agent_id, data in sorted_agents:
            status_icon = "⭐" if agent_id == self.best_agent else "  "
            print(f"{status_icon} Agent {agent_id:<2} "
                  f"{data['timesteps']:<12,} "
                  f"{data['reward']:<12.2f} "
                  f"{data['badges']:<8} "
                  f"{data['exploration']:<10}")
        
        print("\nGlobal Metrics:")
        print(f"  Total Timesteps: {sum(a['timesteps'] for a in self.agents.values()):,}")
        print(f"  Best Reward: {self.best_reward:.2f}")
        print(f"  Global Exploration: {len(self.global_exploration)} coordinates")
        print("=" * 80)
    
    def run(self):
        """Main coordination loop."""
        print("\n🚀 Starting swarm coordination...")
        print("Agents will train independently and sync periodically")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                # In real implementation, this would:
                # 1. Monitor agent processes
                # 2. Collect metrics from shared files
                # 3. Trigger synchronization when needed
                # 4. Aggregate exploration data
                
                time.sleep(10)  # Check every 10 seconds
                
                # Simulate metric updates (in production, read from agent files)
                for agent_id in range(self.num_agents):
                    if agent_id not in self.agents:
                        self.register_agent(agent_id)
                    
                    # Simulated metrics (replace with actual file reading)
                    # In production: read from sessions/agent_X/metrics.json
                    pass
                
                # Check if sync needed
                if self.should_sync():
                    self.aggregate_exploration()
                    self.synchronize_weights()
                    self.print_status()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Coordinator stopped by user")
            self.save_swarm_stats()
            print(f"\n📁 Final statistics saved to: {self.metrics_dir}")

# ============================================================================
# AGENT WRAPPER - Modified training with swarm support
# ============================================================================

class SwarmAgent:
    """
    Wrapper for training agent with swarm coordination.
    
    This agent:
    - Trains independently with PPO
    - Periodically reports metrics to coordinator
    - Loads shared weights when available
    - Shares exploration data
    """
    
    def __init__(self, agent_id, coordinator_path, config):
        self.agent_id = agent_id
        self.coordinator_path = Path(coordinator_path)
        self.config = config
        
        # Create agent-specific directory
        self.agent_path = self.coordinator_path / f"agent_{agent_id}"
        self.agent_path.mkdir(exist_ok=True)
        
        # Metrics file
        self.metrics_file = self.agent_path / 'metrics.json'
        
        # Exploration file
        self.exploration_file = self.coordinator_path / 'exploration' / f'agent_{agent_id}_coords.json'
        
        print(f"🤖 Agent {agent_id} initialized")
        print(f"   Path: {self.agent_path}")
    
    def save_metrics(self, metrics):
        """Save current metrics for coordinator."""
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_exploration(self, coordinates):
        """Save exploration data for sharing."""
        # Convert set to list for JSON serialization
        coords_list = list(coordinates)
        with open(self.exploration_file, 'w') as f:
            json.dump(coords_list, f)
    
    def check_for_sync(self):
        """Check if new weights are available from coordinator."""
        sync_file = self.coordinator_path / 'shared_weights' / 'latest_sync.json'
        
        if sync_file.exists():
            try:
                with open(sync_file, 'r') as f:
                    sync_info = json.load(f)
                
                # Check if this is a new sync
                sync_id = sync_info['sync_id']
                if not hasattr(self, 'last_sync_id') or sync_id > self.last_sync_id:
                    self.last_sync_id = sync_id
                    return sync_info
            except Exception as e:
                print(f"⚠️  Error checking sync: {e}")
        
        return None
    
    def load_shared_weights(self, model_path):
        """Load shared weights from best agent."""
        print(f"\n🔄 Agent {self.agent_id}: Loading shared weights from {model_path}")
        # In production: model.load(model_path)
        return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Create coordinator
    coordinator = SwarmCoordinator(
        num_agents=args.agents,
        sync_frequency=args.sync_freq,
        strategy=args.strategy,
        session_dir=args.session_dir
    )
    
    print("\n" + "=" * 80)
    print("SWARM LEARNING SETUP")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"  Agents: {args.agents}")
    print(f"  Sync Frequency: {args.sync_freq:,} timesteps")
    print(f"  Strategy: {args.strategy}")
    print(f"\n🔧 To integrate with training:")
    print(f"  1. Modify pokemon_red_training.py to use SwarmAgent wrapper")
    print(f"  2. Set ENV_CONFIG['swarm_mode'] = True")
    print(f"  3. Set ENV_CONFIG['coordinator_path'] = '{coordinator.swarm_path}'")
    print(f"  4. Run multiple training instances with different agent IDs")
    
    print("\n🚀 Starting coordinator...")
    print("   Monitor swarm progress in real-time")
    print("   Statistics saved to:", coordinator.metrics_dir)
    
    # Run coordination loop
    coordinator.run()