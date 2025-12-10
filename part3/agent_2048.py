import abc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections

class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, observation, action_mask=None):
        """Select an action based on observation."""
        pass

    def save(self, path):
        """Save agent state."""
        pass

    def load(self, path):
        """Load agent state."""
        pass

class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, action_mask=None):
        if action_mask is not None:
             valid_actions = np.where(action_mask)[0]
             if len(valid_actions) > 0:
                 return int(random.choice(valid_actions))
        return self.action_space.sample()

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNAgent(BaseAgent):
    def __init__(self, grid_size, action_space_n, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.grid_size = grid_size
        self.input_dim = grid_size * grid_size
        self.action_space_n = action_space_n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net = QNetwork(self.input_dim, action_space_n).to(self.device)
        self.target_net = QNetwork(self.input_dim, action_space_n).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.target_update_freq = 100
        self.steps_done = 0

    def preprocess(self, observation):
        # Flatten and log2 transform
        # Add a small epsilon to avoid log2(0)
        # obs is numpy array
        obs_flat = observation.flatten()
        # Use log2 to scale values: log2(0+1)=0, log2(2)=1, log2(4)=2...
        obs_log = np.log2(np.maximum(obs_flat, 1))
        return torch.FloatTensor(obs_log).unsqueeze(0).to(self.device)

    def act(self, observation, action_mask=None):
        if random.random() < self.epsilon:
            if action_mask is not None:
                # Sample from valid actions
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return int(random.choice(valid_actions))
            return random.randint(0, self.action_space_n - 1)
        else:
            with torch.no_grad():
                state_tensor = self.preprocess(observation)
                q_values = self.q_net(state_tensor)
                
                if action_mask is not None:
                    # Mask invalid actions with -inf
                    # Convert mask to boolean tensor
                    mask_tensor = torch.BoolTensor(action_mask).to(self.device)
                    # Use a very large negative number instead of -inf to avoid NaN if all are invalid (though unlikely)
                    q_values[0, ~mask_tensor] = -1e9
                
                return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        # states is list of numpy arrays, need to preprocess each
        states_tensor = torch.cat([self.preprocess(s) for s in states])
        next_states_tensor = torch.cat([self.preprocess(s) for s in next_states])
        
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Q(s, a)
        q_values = self.q_net(states_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Double DQN:
        # Action selection: argmax_a Q(s', a; theta)  (using online network)
        # Evaluation: Q(s', a_max; theta_target)      (using target network)
        
        with torch.no_grad():
            # Get best action for next state from online network
            next_state_actions = self.q_net(next_states_tensor).argmax(1).unsqueeze(1)
            # Evaluate that action using target network
            next_q_values = self.target_net(next_states_tensor)
            next_q_value = next_q_values.gather(1, next_state_actions).squeeze(1)
            
            expected_q_value = rewards_tensor + self.gamma * next_q_value * (1 - dones_tensor)
            
        loss = nn.MSELoss()(q_value, expected_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
