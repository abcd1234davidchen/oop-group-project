import abc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BaseAgent(abc.ABC):
    '''Abstract Base Class for Agents in 2048 environment.'''
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
    '''Random Agent for 2048 environment. Serve as a baseline.'''
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, observation, action_mask=None):
        if action_mask is not None:
             valid_actions = np.where(action_mask)[0]
             if len(valid_actions) > 0:
                 return int(random.choice(valid_actions))
        return self.action_space.sample()

class QNetwork(nn.Module):
    '''Neural Network for approximating Q-values. Inherits from nn.Module.'''
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        
        # Convolutional layers to process the 2048 board
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    '''Experience Replay Buffer for storing and sampling experiences.'''
    def __init__(self, capacity, state_shape=None, state_dtype=np.int32):
        # Initialize the replay buffer with given capacity and state shape/dtype
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape) if state_shape is not None else None
        self.state_dtype = state_dtype
        self.position = 0
        self.size = 0
        self._initialized = False

        if self.state_shape is not None:
            self._init_arrays(self.state_shape, self.state_dtype)

    def _init_arrays(self, state_shape, state_dtype):
        # Initialize numpy arrays for states, actions, rewards, next_states, and dones
        self.states = np.zeros((self.capacity,) + tuple(state_shape), dtype=state_dtype)
        self.next_states = np.zeros((self.capacity,) + tuple(state_shape), dtype=state_dtype)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.uint8)
        self._initialized = True

    def push(self, state, action, reward, next_state, done):
        # Store a new experience in the buffer
        s = np.asarray(state)
        ns = np.asarray(next_state)

        # Initialize arrays if not already done
        if not self._initialized:
            self._init_arrays(s.shape, s.dtype)

        self.states[self.position] = s
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = ns
        self.dones[self.position] = 1 if done else 0

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        idx = np.random.choice(self.size, int(batch_size), replace=False)
        return (
            self.states[idx].copy(),
            self.actions[idx].copy(),
            self.rewards[idx].copy(),
            self.next_states[idx].copy(),
            self.dones[idx].copy(),
        )

    def __len__(self):
        # Return the current size of the buffer
        return int(self.size)

class DQNAgent(BaseAgent):
    '''Agent implementing Deep Q-Network for 2048 environment.'''
    def __init__(self, grid_size, action_space_n, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        # Initialize the DQN agent with environment parameters
        self.grid_size = grid_size
        self.input_dim = grid_size * grid_size
        self.action_space_n = action_space_n
        
        # Gamma is for discounting future rewards, Epsilon parameters for exploration
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Set device for computation
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize Q-Network and Target Network
        self.q_net = QNetwork(self.input_dim, action_space_n).to(self.device)
        self.target_net = QNetwork(self.input_dim, action_space_n).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer, replay buffer, and training parameters
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(50000, state_shape=(grid_size, grid_size), state_dtype=np.int32)
        self.batch_size = 512
        self.target_update_freq = 100
        self.steps_done = 0

    def preprocess(self, observation):
        # Preprocess the observation into a tensor suitable for the network
        tensor = np.log2(np.maximum(observation, 1))
        tensor = torch.from_numpy(tensor).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)  # Add channel dimension
        # Now tensor shape is (batch_size, 1, grid_size, grid_size)
        return tensor.to(self.device)

    def act(self, observation, action_mask=None):
        # Handle single observation case
        if observation.ndim == 2:
            observation = np.expand_dims(observation, axis=0)
            if action_mask is not None:
                action_mask = np.expand_dims(action_mask, axis=0)
        
        # Determine batch size and exploration decisions
        batch_size = len(observation)
        explores = np.random.random(batch_size) < self.epsilon

        # Get greedy actions from Q-Network
        with torch.no_grad():
            self.q_net.eval()
            state_tensor = self.preprocess(observation)
            q_values = self.q_net(state_tensor)

            # Apply action mask if provided
            if action_mask is not None:
                mask_tensor = torch.BoolTensor(action_mask).to(self.device)
                q_values[~mask_tensor] = -1e9

            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            self.q_net.train()

        # Combine greedy and random actions based on exploration
        actions = greedy_actions.copy()
        if np.any(explores):
            for i in np.where(explores)[0]:
                if action_mask is not None:
                    valid_choices = np.where(action_mask[i])[0]
                    if len(valid_choices) > 0:
                        actions[i] = np.random.choice(valid_choices)
                    else:
                        actions[i] = np.random.randint(0, self.action_space_n)
                else:
                    actions[i] = np.random.randint(0, self.action_space_n)
        return actions if len(actions) > 1 else int(actions[0])

    def update(self):
        # Update only when enough samples are available
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_tensor = self.preprocess(np.array(states))
        next_states_tensor = self.preprocess(np.array(next_states))
        
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float, device=self.device)

        # Q(s, a)
        q_values = self.q_net(states_tensor)
        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # Get best action for next state from training network
            next_state_actions = self.q_net(next_states_tensor).argmax(1).unsqueeze(1)
            # Evaluate that action using target network
            next_q_values = self.target_net(next_states_tensor)
            next_q_value = next_q_values.gather(1, next_state_actions).squeeze(1)
            
            expected_q_value = rewards_tensor + self.gamma * next_q_value * (1 - dones_tensor)
            
        loss = nn.SmoothL1Loss()(q_value, expected_q_value)
        
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
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)

    def save(self, path):
        # Save the Q-Network state
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        # Load the Q-Network state
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
