# Fixed simulation/simulation_core.py

import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

n = 50
density_field = ti.field(dtype=ti.f32, shape=(n,))
velocity_field = ti.field(dtype=ti.f32, shape=(n,))

traffic_lights = [10, 25, 40]
light_states = {10: 1, 25: 1, 40: 1}

@ti.kernel
def update_flow():
    for i in range(n - 1):
        skip = False
        for light in ti.static(traffic_lights):
            if i == light and light_states[light] == 0:
                skip = True
        if skip:
            continue

        flow = ti.min(density_field[i], 1.0)
        density_field[i] -= flow
        density_field[i + 1] += flow

def set_traffic_lights(positions, states):
    """Set traffic light states - fixed to handle list inputs properly"""
    if isinstance(positions, list) and isinstance(states, list):
        for pos, state in zip(positions, states):
            if pos in light_states:
                light_states[pos] = state
    else:
        # Handle legacy format
        for idx, pos in enumerate(positions):
            if pos in light_states and idx < len(states):
                light_states[pos] = states[idx]

def get_current_state():
    """Get current simulation state for RL agent"""
    current_density = density_field.to_numpy()
    avg_density = np.mean(current_density)
    max_density = np.max(current_density)
    min_density = np.min(current_density)
    std_density = np.std(current_density)
    return [avg_density, max_density, min_density, std_density]

def run_simulation(steps=100, density=None, return_history=False):
    """Enhanced simulation runner with better state management"""
    if density is not None:
        init_density(density)
    
    results = []
    states = []  # Track states for RL
    
    for step in range(steps):
        # Store current state before update
        if return_history:
            states.append(get_current_state())
        
        update_flow()
        results.append(density_field.to_numpy().copy())
    
    if return_history:
        return np.array(results), np.array(states)
    else:
        return np.array(results)

def init_density(density_np):
    """Initialize density field with proper bounds checking"""
    max_index = min(len(density_np), n)
    for i in range(max_index):
        density_field[i] = max(0.0, min(1.0, density_np[i]))  # Clamp values
    for i in range(max_index, n):
        density_field[i] = 0.0

def reset_simulation():
    """Reset simulation to initial state"""
    for i in range(n):
        density_field[i] = 0.0
        velocity_field[i] = 0.0
    
    # Reset traffic lights to default green
    for light in traffic_lights:
        light_states[light] = 1