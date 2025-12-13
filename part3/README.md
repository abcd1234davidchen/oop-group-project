# Part 3: 2048 Reinforcement Learning Agent

This directory contains a custom implementation of the 2048 game, wrapped as a Gymnasium environment, and a Deep Q-Network (DQN) agent trained to play it.

## 📂 File Structure

- **`main_2048.py`**: The entry point for the project. Handles training the DQN agent, testing the trained agent, and running a random agent baseline.
- **`agent_2048.py`**: Contains the implementations of:
  - `DQNAgent`: A Double DQN agent with a Replay Buffer.
  - `RandomAgent`: A baseline agent that selects random valid moves.
  - `QNetwork`: The PyTorch neural network used for Q-value approximation.
- **`env_2048.py`**: A custom [Gymnasium](https://gymnasium.farama.org/) environment (`2048-v0`) that wraps the game logic. It provides the standard `reset`, `step`, and `render` interface.
- **`game_2048.py`**: The core logic for the 2048 game, including movement, merging rules, and Pygame-based rendering.

## ⚙️ Installation

Ensure you have the required dependencies installed. You can install them using pip:

```bash
pip install gymnasium numpy torch pygame tqdm
```

## 🚀 Usage

You can run the project using `main_2048.py` with various command-line arguments.

### 1. Train the Agent
To train the DQN agent from scratch:

```bash
# Train for 10,000 episodes (default) and save to dqn_2048.pth
python main_2048.py --action train

# Train for a custom number of episodes
python main_2048.py --action train --train_episodes 500
```
*The model will be saved as `dqn_2048.pth` after training.*

### 2. Test the Trained Agent
To watch the trained agent play:

```bash
# Run 5 test episodes with rendering enabled
python main_2048.py --action test --test_episodes 5
```
*Note: This requires a `dqn_2048.pth` file to exist (created during training).*

### 3. Run Random Agent Baseline
To see how a random agent performs:

```bash
python main_2048.py --action random
```

### 4. Rendering
By default, training runs without rendering to speed up the process. You can enable rendering (human mode) by adding the `--render True` flag, though this will significantly slow down training. Testing always enables rendering.

```bash
python main_2048.py --action train --render True
```

## 🧠 Approach

### State Representation
The grid (4x4) is log2-transformed to normalize the input values (e.g., 2 -> 1, 4 -> 2, ..., 2048 -> 11). This helps the neural network learn more effectively compared to using raw tile values.

### Reward Function
The environment provides rewards based on:
- **Merge Score**: The value of the new tile created by merging (e.g., merging two 4s yields +8).
- **Penalties/Bonuses**: Logic to penalize clutter (low empty cells) and reward strategic placement (e.g., keeping the max tile in a corner) is implemented in `env_2048.py`.

### Agent (DQN) Implementation Details

The agent is implemented in `agent_2048.py` using PyTorch. It employs a **Double Deep Q-Network (Double DQN)** algorithm to stabilize training.

#### Network Architecture (`QNetwork`)
A fully connected feed-forward network:
- **Input Layer**: 16 neurons (flattened 4x4 grid).
- **Hidden Layer 1**: 256 neurons, ReLU activation.
- **Hidden Layer 2**: 256 neurons, ReLU activation.
- **Output Layer**: 4 neurons (corresponding to valid actions: Left, Up, Right, Down).

#### Training Hyperparameters
- **Optimizer**: Adam (`lr=1e-3`)
- **Loss Function**: Mean Squared Error (MSE)
- **Discount Factor ($\gamma$)**: 0.99
- **Batch Size**: 64
- **Replay Buffer Size**: 10,000 transitions
- **Target Network Update Frequency**: Every 100 steps
- **Exploration (Epsilon-Greedy)**:
  - Starts at $\epsilon=1.0$
  - Decays by factor `0.999` per episode (configurable in `main_2048.py`)
  - Minimum $\epsilon=0.01$

#### Preprocessing
Before being fed into the network, the raw grid values (powers of 2) are **log-transformed** using $log_2(x)$ to reduce the scale variance (e.g., 2048 becomes 11). This helps the neural network converge faster.
