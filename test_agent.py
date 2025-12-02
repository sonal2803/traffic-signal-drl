# Training and Comparison Script: PSO vs Adam

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

from rl_agent.environment import FourWayIntersectionEnv
from rl_agent.agent import DQNAgent as PSOAgent  # Your PSO agent
from rl_agent.adam_agent import AdamDQNAgent  # New Adam agent

class OptimizerComparison:
    def __init__(self):
        self.pso_results = []
        self.adam_results = []
        self.training_history = {
            'pso': {'rewards': [], 'losses': [], 'densities': [], 'times': []},
            'adam': {'rewards': [], 'losses': [], 'densities': [], 'times': []}
        }
        
    def train_agent(self, agent, env, episodes=100, agent_type="PSO"):
        """Train an agent and collect metrics"""
        print(f"\n🚀 Training {agent_type} Agent for {episodes} episodes...")
        
        episode_rewards = []
        episode_losses = []
        episode_densities = []
        episode_times = []
        
        for episode in range(episodes):
            start_time = time.time()
            
            state = env.reset()
            total_reward = 0
            episode_loss = []
            
            for step in range(env.max_steps):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                # Train the agent
                if len(agent.memory) > 32:  # Minimum batch size
                    if agent_type == "PSO":
                        loss = agent.replay(batch_size=16)  # Smaller batch for PSO
                    else:
                        loss = agent.replay(batch_size=32)
                    
                    if loss is not None:
                        episode_loss.append(loss)
                
                if done:
                    break
            
            # Update target network for Adam agent
            if agent_type == "Adam" and episode % 10 == 0:
                agent.update_target_model()
            
            episode_time = time.time() - start_time
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            
            episode_rewards.append(total_reward)
            episode_losses.append(avg_loss)
            episode_densities.append(env.get_episode_stats().get('avg_queue', 0))
            episode_times.append(episode_time)
            
            # Progress reporting
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_density = np.mean(episode_densities[-20:])
                print(f"{agent_type} Episode {episode+1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Avg Density: {avg_density:.3f}, "
                      f"Epsilon: {agent.epsilon:.3f}, "
                      f"Time: {episode_time:.2f}s")
        
        return {
            'rewards': episode_rewards,
            'losses': episode_losses,
            'densities': episode_densities,
            'times': episode_times
        }
    
    def run_comparison(self, episodes=200):
        """Run complete comparison between PSO and Adam"""
        print("=" * 60)
        print("🔬 STARTING PSO vs ADAM OPTIMIZER COMPARISON")
        print("=" * 60)
        
        # Create environments
        env_pso = FourWayIntersectionEnv()
        env_adam = FourWayIntersectionEnv()
        
        # Initialize agents
        pso_agent = PSOAgent(state_size=4, action_size=2)
        adam_agent = AdamDQNAgent(state_size=4, action_size=2, learning_rate=0.0005)
        
        # Train PSO agent
        pso_results = self.train_agent(pso_agent, env_pso, episodes, "PSO")
        self.training_history['pso'] = pso_results
        
        # Train Adam agent
        adam_results = self.train_agent(adam_agent, env_adam, episodes, "Adam")
        self.training_history['adam'] = adam_results
        
        # Generate comparison report
        self.generate_report()
        self.plot_comparison()
        
        return pso_agent, adam_agent
    
    def generate_report(self):
        """Generate detailed comparison report"""
        print("\n" + "=" * 60)
        print("📊 COMPARISON RESULTS")
        print("=" * 60)
        
        pso_data = self.training_history['pso']
        adam_data = self.training_history['adam']
        
        # Final performance metrics
        pso_final_reward = np.mean(pso_data['rewards'][-20:])
        adam_final_reward = np.mean(adam_data['rewards'][-20:])
        
        pso_final_density = np.mean(pso_data['densities'][-20:])
        adam_final_density = np.mean(adam_data['densities'][-20:])
        
        pso_avg_time = np.mean(pso_data['times'])
        adam_avg_time = np.mean(adam_data['times'])
        
        print(f"📈 FINAL PERFORMANCE (Last 20 episodes avg):")
        print(f"   PSO  - Reward: {pso_final_reward:.4f}, Density: {pso_final_density:.4f}")
        print(f"   Adam - Reward: {adam_final_reward:.4f}, Density: {adam_final_density:.4f}")
        
        print(f"\n⏱️  TRAINING TIME:")
        print(f"   PSO  - Avg per episode: {pso_avg_time:.2f}s")
        print(f"   Adam - Avg per episode: {adam_avg_time:.2f}s")
        
        print(f"\n🎯 CONVERGENCE:")
        pso_convergence = self.find_convergence(pso_data['rewards'])
        adam_convergence = self.find_convergence(adam_data['rewards'])
        print(f"   PSO  - Converged at episode: {pso_convergence}")
        print(f"   Adam - Converged at episode: {adam_convergence}")
        
        print(f"\n🏆 WINNER:")
        if pso_final_reward > adam_final_reward:
            print(f"   PSO wins with {(pso_final_reward - adam_final_reward):.4f} higher reward!")
        elif adam_final_reward > pso_final_reward:
            print(f"   Adam wins with {(adam_final_reward - pso_final_reward):.4f} higher reward!")
        else:
            print("   It's a tie!")
        
        # Save detailed results
        self.save_results()
    
    def find_convergence(self, rewards, window=20, threshold=0.01):
        """Find episode where rewards converged (low variance)"""
        if len(rewards) < window * 2:
            return len(rewards)
        
        for i in range(window, len(rewards) - window):
            recent_std = np.std(rewards[i:i+window])
            if recent_std < threshold:
                return i
        return len(rewards)
    
    def plot_comparison(self):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PSO vs Adam Optimizer Comparison', fontsize=16, fontweight='bold')
        
        pso_data = self.training_history['pso']
        adam_data = self.training_history['adam']
        
        # 1. Rewards over time
        axes[0, 0].plot(self.smooth_curve(pso_data['rewards']), label='PSO', color='red', linewidth=2)
        axes[0, 0].plot(self.smooth_curve(adam_data['rewards']), label='Adam', color='blue', linewidth=2)
        axes[0, 0].set_title('Episode Rewards (Smoothed)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training Loss
        axes[0, 1].plot(self.smooth_curve(pso_data['losses']), label='PSO', color='red', linewidth=2)
        axes[0, 1].plot(self.smooth_curve(adam_data['losses']), label='Adam', color='blue', linewidth=2)
        axes[0, 1].set_title('Training Loss (Smoothed)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Traffic Density
        axes[1, 0].plot(self.smooth_curve(pso_data['densities']), label='PSO', color='red', linewidth=2)
        axes[1, 0].plot(self.smooth_curve(adam_data['densities']), label='Adam', color='blue', linewidth=2)
        axes[1, 0].set_title('Average Queue Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Queue Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training Time per Episode
        axes[1, 1].boxplot([pso_data['times'], adam_data['times']], 
                          labels=['PSO', 'Adam'],
                          patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[1, 1].set_title('Training Time Distribution')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'pso_vs_adam_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def smooth_curve(self, data, window=10):
        """Apply moving average smoothing"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def save_results(self):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'pso_vs_adam_results_{timestamp}.json'
        
        results = {
            'timestamp': timestamp,
            'pso_results': {
                'final_avg_reward': float(np.mean(self.training_history['pso']['rewards'][-20:])),
                'final_avg_density': float(np.mean(self.training_history['pso']['densities'][-20:])),
                'avg_training_time': float(np.mean(self.training_history['pso']['times'])),
                'convergence_episode': self.find_convergence(self.training_history['pso']['rewards'])
            },
            'adam_results': {
                'final_avg_reward': float(np.mean(self.training_history['adam']['rewards'][-20:])),
                'final_avg_density': float(np.mean(self.training_history['adam']['densities'][-20:])),
                'avg_training_time': float(np.mean(self.training_history['adam']['times'])),
                'convergence_episode': self.find_convergence(self.training_history['adam']['rewards'])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def main():
    """Main function to run the comparison"""
    comparison = OptimizerComparison()
    pso_agent, adam_agent = comparison.run_comparison(episodes=150)
    
    # Save trained models
    pso_agent.save_model('pso_traffic_model.pth')
    adam_agent.save_model('adam_traffic_model.pth')
    
    print("\n✅ Comparison complete! Models saved.")
    return comparison

if __name__ == "__main__":
    comparison = main()