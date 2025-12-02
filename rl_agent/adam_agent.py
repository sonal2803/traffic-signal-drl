# Adam-based DQN Agent for comparison

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class AdamDQNAgent:
    def __init__(self, state_size=4, action_size=3, learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = learning_rate
        
        # Neural networks
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        
        # Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training statistics
        self.training_losses = []
        self.episode_rewards = []
        self.q_values_history = []
        
        # Update target network initially
        self.update_target_model()
        
        print(f"✅ Adam Agent initialized with learning rate: {learning_rate}")

    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return int(np.random.choice([0, 1]))  # random phase: 0 or 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            self.q_values_history.append(q_values.squeeze().numpy())
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
        # Keep memory size manageable
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self, batch_size=32):
        """Train the model using experience replay (Double DQN)"""
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Double DQN: action selection by main model, evaluation by target model
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.training_losses.append(loss.item())
        return loss.item()

    def get_training_stats(self):
        """Get training statistics"""
        return {
            'avg_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_q_values': np.mean([np.mean(q) for q in self.q_values_history[-10:]]) if self.q_values_history else 0
        }

    def save_model(self, filepath):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards
        }, filepath)
        print(f"✅ Adam model saved to {filepath}")

    def load_model(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_losses = checkpoint.get('training_losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"✅ Adam model loaded from {filepath}")

    def reset_stats(self):
        """Reset training statistics"""
        self.training_losses = []
        self.episode_rewards = []
        self.q_values_history = []