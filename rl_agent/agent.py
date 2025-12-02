import random
import numpy as np
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)

class PSOOptimizer:
    def __init__(self, model, num_particles=20, w=0.729, c1=1.494, c2=1.494, lr=0.001):
        """
        Particle Swarm Optimization for neural network parameters
        
        Args:
            model: PyTorch neural network model
            num_particles: Number of particles in the swarm
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            lr: Learning rate scaling factor
        """
        self.model = model
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lr = lr
        
        # Get model parameters as a flat vector
        self.param_shapes = []
        self.param_sizes = []
        total_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                shape = param.shape
                size = param.numel()
                self.param_shapes.append(shape)
                self.param_sizes.append(size)
                total_params += size
        
        self.total_params = total_params
        
        # Initialize particles
        self.particles = np.random.randn(num_particles, total_params) * 0.1
        self.velocities = np.zeros((num_particles, total_params))
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(num_particles, float('inf'))
        self.global_best = self.particles[0].copy()
        self.global_best_fitness = float('inf')
        
        print(f"✅ PSO Optimizer initialized with {num_particles} particles and {total_params} parameters!")

    def params_to_vector(self):
        """Convert model parameters to flat vector"""
        vector = []
        for param in self.model.parameters():
            if param.requires_grad:
                vector.append(param.data.flatten())
        return torch.cat(vector).cpu().numpy()

    def vector_to_params(self, vector):
        """Convert flat vector back to model parameters"""
        idx = 0
        for param in self.model.parameters():
            if param.requires_grad:
                size = param.numel()
                param.data = torch.tensor(
                    vector[idx:idx + size].reshape(param.shape),
                    dtype=param.dtype,
                    device=param.device
                )
                idx += size

    def evaluate_fitness(self, states, actions, targets):
        """Evaluate fitness of current model parameters"""
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(states))
            loss = nn.MSELoss()(predictions, torch.FloatTensor(targets))
            return loss.item()

    def step(self, states, actions, targets):
        """Perform one PSO optimization step"""
        # Evaluate all particles
        current_params = self.params_to_vector()
        
        for i in range(self.num_particles):
            # Set model parameters to particle position
            self.vector_to_params(self.particles[i])
            
            # Evaluate fitness
            fitness = self.evaluate_fitness(states, actions, targets)
            
            # Update personal best
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best[i] = self.particles[i].copy()
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()
        
        # Update velocities and positions
        for i in range(self.num_particles):
            r1 = np.random.random(self.total_params)
            r2 = np.random.random(self.total_params)
            
            # Update velocity
            self.velocities[i] = (
                self.w * self.velocities[i] +
                self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                self.c2 * r2 * (self.global_best - self.particles[i])
            )
            
            # Update position
            self.particles[i] += self.lr * self.velocities[i]
        
        # Set model to best parameters
        self.vector_to_params(self.global_best)
        
        return self.global_best_fitness

    def zero_grad(self):
        """Compatibility method (PSO doesn't use gradients)"""
        pass

class DQNAgent:
    def __init__(self, state_size=4, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = DQN(state_size, action_size)
        
        # Replace Adam optimizer with PSO optimizer
        self.optimizer = PSOOptimizer(
            model=self.model,
            num_particles=10,  # Fewer particles
            w=0.8,             # Slightly higher inertia
            c1=1.2,            # Lower cognitive
            c2=1.2,            # Lower social
            lr=0.005           # Lower learning rate
        )
        
        self.criterion = nn.MSELoss()
        print("✅ Agent initialized with PSO optimizer!")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return int(np.random.choice([0, 1]))  # random phase: 0 or 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=16):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        
        # Prepare batch data for PSO
        states = []
        targets = []
        
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0)
                target_value = reward
                if not done:
                    target_value = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()

                target_q = q_values.clone().detach()
                target_q[action] = torch.tensor(float(target_value), dtype=torch.float32)
            
            states.append(state)
            targets.append(target_q.numpy())
        
        # Use PSO to optimize model parameters
        states = np.array(states)
        targets = np.array(targets)
        
        # Perform PSO optimization step
        fitness = self.optimizer.step(states, None, targets)
        
        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return fitness

    def get_best_fitness(self):
        """Get the current best fitness from PSO"""
        return self.optimizer.global_best_fitness

    def save_model(self, filepath):
        """Save the model with best PSO parameters"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pso_best_params': self.optimizer.global_best,
            'epsilon': self.epsilon
        }, filepath)

    def load_model(self, filepath):
        """Load the saved model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = checkpoint['epsilon']
        if 'pso_best_params' in checkpoint:
            self.optimizer.vector_to_params(checkpoint['pso_best_params'])