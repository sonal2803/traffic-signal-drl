import numpy as np
from typing import Tuple, List, Dict, Optional

class FourWayIntersectionEnv:
    """
    Four-way intersection traffic environment.
    Each direction (N, S, E, W) has a queue.
    Two-phase control: NS green/EW red or EW green/NS red.
    """

    def __init__(self, max_steps: int = 200):
        self.directions = ['north', 'south', 'east', 'west']
        self.num_dirs = 4
        self.max_steps = max_steps
        self.current_step = 0

        # Queues: number of cars waiting in each direction
        self.queues = np.zeros(self.num_dirs, dtype=np.float32)

        # 0 = NS green, EW red; 1 = EW green, NS red
        self.phase = 0
        self.phase_durations = [0, 0]  # How long each phase has been active
        self.forced_switch = [0, 0]    # Forced switch counters for each phase
        self.min_green_duration = 60  # minimum green duration in steps
        self.current_green_time = 0

        # Statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'avg_queue': 0.0,
            'max_queue': 0.0,
            'throughput': 0.0
        }

        print("✅ FourWayIntersectionEnv initialized.")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.queues = np.zeros(self.num_dirs, dtype=np.float32)
        self.phase = 0
        self.phase_durations = [0, 0]
        self.forced_switch = [0, 0]
        self.current_green_time = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'avg_queue': 0.0,
            'max_queue': 0.0,
            'throughput': 0.0
        }
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        action: 0 = NS green, EW red; 1 = EW green, NS red
        """
        self.current_step += 1

        # Only allow phase change if min green duration is met
        if self.phase == action:
            self.current_green_time += 1
        else:
            if self.current_green_time >= self.min_green_duration:
                self.phase = action
                self.current_green_time = 1
            else:
                # Ignore agent's action, keep current phase
                self.current_green_time += 1

        # Forced switch logic (still applies for max duration)
        if self.phase_durations[self.phase] >= 120 or self.forced_switch[self.phase] > 0:
            self.forced_switch[self.phase] = 45
            self.phase = 1 - self.phase  # Switch phase
            for k in range(2):
                if self.forced_switch[k] > 0:
                    self.forced_switch[k] -= 1
            self.phase_durations = [0, 0]
        else:
            self.phase_durations[self.phase] += 1
            self.phase_durations[1 - self.phase] = 0

        # Simulate traffic flow
        self._simulate_traffic_flow()

        # Calculate reward
        reward = self._calculate_reward()

        # Update statistics
        self.episode_stats['total_reward'] += reward
        self.episode_stats['avg_queue'] = float(np.mean(self.queues))
        self.episode_stats['max_queue'] = float(np.max(self.queues))

        done = self.current_step >= self.max_steps
        return self._get_state(), reward, done

    def _simulate_traffic_flow(self):
        # Inflow: lower mean inflow for easier learning
        inflow = np.random.poisson(1, size=self.num_dirs)
        self.queues += inflow
        # Outflow: cars leave if their direction is green
        if self.phase == 0:
            for i in [0, 1]:
                out = min(self.queues[i], 5)
                self.queues[i] -= out
                self.episode_stats['throughput'] += out
        else:
            for i in [2, 3]:
                out = min(self.queues[i], 5)
                self.queues[i] -= out
                self.episode_stats['throughput'] += out
        # Cap queue length
        self.queues = np.clip(self.queues, 0, 30)

    def _calculate_reward(self) -> float:
        queue_penalty = -2.0 * np.sum(self.queues)
        switch_penalty = -20 if getattr(self, 'last_phase', self.phase) != self.phase else 0
        throughput_reward = self.episode_stats['throughput'] * 0.1
        self.last_phase = self.phase
        return float(queue_penalty + switch_penalty + throughput_reward)

    def _get_state(self) -> np.ndarray:
        # State: queue lengths for all directions
        return self.queues.copy()

    def get_episode_stats(self) -> Dict:
        return self.episode_stats.copy()

    def render(self):
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print(f"Queues: N={self.queues[0]:.1f}, S={self.queues[1]:.1f}, E={self.queues[2]:.1f}, W={self.queues[3]:.1f}")
        print(f"Phase: {'NS green' if self.phase == 0 else 'EW green'}")
        print(f"Avg queue: {self.episode_stats['avg_queue']:.2f}, Max queue: {self.episode_stats['max_queue']:.2f}")


# Example usage and testing
if __name__ == "__main__":
    # Test the environment
    env = FourWayIntersectionEnv()
    
    print("Testing FourWayIntersectionEnv...")
    state = env.reset()
    print(f"Initial state: {state}")
    
    # Test a few steps
    for step in range(5):
        # Random action: 0 or 1
        action = np.random.choice([0, 1])
        next_state, reward, done = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"Action: {action}")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward:.3f}")
        print(f"Done: {done}")
        
        env.render()
        
        if done:
            break
    
    print("\n✅ Environment test completed!")