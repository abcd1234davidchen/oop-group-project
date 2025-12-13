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

# I need further examination on conv portion
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

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
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Overwrite memory at the current pointer (Circular Buffer)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Random access in a list is INSTANT (O(1))
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
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.q_net = QNetwork(self.input_dim, action_space_n).to(self.device)
        self.target_net = QNetwork(self.input_dim, action_space_n).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(50000)
        self.batch_size = 1024
        self.target_update_freq = 100
        self.steps_done = 0

    def preprocess(self, observation):
        # Flatten and log2 transform
        # Add a small epsilon to avoid log2(0)
        # obs is numpy array
        # Use log2 to scale values: log2(0+1)=0, log2(2)=1, log2(4)=2...
        obs_log = np.log2(np.maximum(observation, 1))
        tensor = torch.FloatTensor(obs_log).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

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
        
    def act_batch(self, observations, action_masks=None):
        """
        Selects actions for a batch of environments (Vectorized).
        
        Args:
            observations: Numpy array of shape (Batch_Size, 4, 4)
            action_masks: Optional Numpy array of shape (Batch_Size, 4)
            
        Returns:
            Numpy array of shape (Batch_Size,) containing actions (int)
        """
        batch_size = len(observations)
        actions = np.zeros(batch_size, dtype=int)
        
        # 1. Determine which agents will explore (Random) vs exploit (Greedy)
        # We generate a list of random numbers, one for each game
        explore_flags = np.random.random(batch_size) < self.epsilon
        
        # --- PATH A: GREEDY ACTIONS (The Neural Network) ---
        # We run the network for EVERYONE first (it's faster to batch 32 than split them)
        with torch.no_grad():
            self.q_net.eval()
            
            # Preprocess the whole batch at once: (32, 4, 4) -> (32, 1, 4, 4)
            obs_log = np.log2(np.maximum(observations, 1))
            obs_log = np.expand_dims(obs_log, axis=1)
            state_tensor = torch.FloatTensor(obs_log).to(self.device)
            
            # Forward Pass
            q_values = self.q_net(state_tensor) # Shape: (32, 4)
            
            # Apply Masks (if provided)
            if action_masks is not None:
                # Convert mask to Tensor (True = Valid, False = Invalid)
                mask_tensor = torch.BoolTensor(action_masks).to(self.device)
                # Set invalid actions to -infinity so argmax never picks them
                # Note: We use ~mask_tensor because typically mask=1 means valid
                # If your mask logic is 0=valid, remove the ~
                q_values[~mask_tensor] = -1e9
                
            # Select best actions
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            self.q_net.train()

        # --- PATH B: RANDOM ACTIONS ---
        # For the games that decided to explore, overwrite the greedy action with a random one
        if np.any(explore_flags):
            for i in np.where(explore_flags)[0]:
                if action_masks is not None:
                    # Pick from valid actions only
                    valid_actions = np.where(action_masks[i])[0]
                    if len(valid_actions) > 0:
                        actions[i] = np.random.choice(valid_actions)
                    else:
                        actions[i] = np.random.randint(0, self.action_space_n)
                else:
                    actions[i] = np.random.randint(0, self.action_space_n)
        
        # --- COMBINE ---
        # Copy greedy actions to the result array
        # Then (implicitly) the loop above already overwrote the random ones
        # Actually, simpler logic:
        # 1. Fill 'actions' with greedy results
        actions = greedy_actions.copy()
        
        # 2. Overwrite the 'explore' indices with random choices
        if np.any(explore_flags):
             for i in np.where(explore_flags)[0]:
                if action_masks is not None:
                    valid_choices = np.where(action_masks[i])[0]
                    if len(valid_choices) > 0:
                        actions[i] = np.random.choice(valid_choices)
                    else:
                        actions[i] = np.random.randint(0, self.action_space_n)
                else:
                    actions[i] = np.random.randint(0, self.action_space_n)

        return actions

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        # states is list of numpy arrays, need to preprocess each
        state_batch = np.array(states)
        state_batch = np.log2(np.maximum(state_batch, 1))
        state_batch = np.expand_dims(state_batch, axis=1)

        next_state_batch = np.array(next_states)
        next_state_batch = np.log2(np.maximum(next_state_batch, 1))
        next_state_batch = np.expand_dims(next_state_batch, axis=1)

        states_tensor = torch.FloatTensor(state_batch).to(self.device)
        next_states_tensor = torch.FloatTensor(next_state_batch).to(self.device)
        
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
        self.memory.push(state, action, reward, next_state, done)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
