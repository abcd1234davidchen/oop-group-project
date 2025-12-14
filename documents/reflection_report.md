# OOP Group Project – Part 3 Reflection Report (2048 RL)

**Group ID:** 23

**Members:** 陳展皝、簡才斌、鄭祐丞

## Overview

For Part 3, we went built a 2048 game environment from scratch, and trained a Deep Q-Network agent to control. We structured everything around the Gymnasium Env interface and split the code into separate pieces for the agent, environment, and training/testing scripts.

---

## 1. OOP concept(s) implemented in Part 3

### 1.1 Abstraction (Interfaces / Contracts)

Our key design goal was to define a clean and minimal contract between components.

- **Agent interface (`BaseAgent`)** (`part3/agent_2048.py`): `BaseAgent` is an abstract base class that requires every agent to implement `act(...)`, so training/testing code can call the same method regardless of the underlying strategy.
- **Environment interface (Gymnasium `gym.Env`)** in (`part3/env_2048.py`): Our `TwentyFortyEightEnv` follows the Gymnasium API (`reset`, `step`, `render`, `close`), which means we can drop it straight into `gym.make(...)` and `gym.vector.SyncVectorEnv(...)` without extra glue code.

This meant the training loop could just treat everything as black boxes with guaranteed inputs/outputs.

### 1.2 Inheritance (Reuse + Specialization)

We used inheritance mostly to avoid repeating ourselves.

`RandomAgent` and `DQNAgent` both inherit from `BaseAgent`, so they follow the same contract but differ in how they pick actions (one's random, one's learned). `DQNAgent` also adds `save/load` methods for checkpointing. In addition, `QNetwork` inherits from `nn.Module` and `TwentyFortyEightEnv` inherits from `gym.Env`, which gives us all the PyTorch and Gym machinery for free.

This let us focus on the 2048-specific logic instead of reinventing wheels.

### 1.3 Polymorphism (Interchangeable Components)

Polymorphism appears when we swap different concrete types through the same interface.

In `part3/main_2048.py`, the same `test(...)` function works with both `DQNAgent` and `RandomAgent` because they both have `act(...)`. We also made `act()` accept an optional `action_mask`, so the environment can filter out illegal moves without the training loop needing to know 2048 rules.

This made it really easy to swap in a baseline agent vs. the trained one during experiments.

### 1.4 Encapsulation (State + Behavior bundled together)

Each class keeps its own data private and only exposes methods to interact with it.

`TwentyFortyEightEnv` hides the board state (`self.board`), reward calculations, game-over checks, and rendering behind `reset/step/render`. Similarly, `ReplayBuffer` handles storing transitions and sampling batches internally—external code just calls `push(...)` and `sample(...)`.

This helps prevent accidental bugs where one part of the code messes with another part's data.

### 1.5 Composition & Separation of Concerns (Modular design)

We broke Part 3 into separate modules:

- `env_2048.py`: game mechanics, rewards, action masking, rendering
- `agent_2048.py`: DQN agent, replay buffer, neural network
- `main_2048.py`: training/testing orchestration

This means we can tweak the reward function without touching the neural network, or swap in a different agent algorithm without rewriting the environment.

---

## 2. Implementation of Part 3

### 2.1 Custom 2048 Gymnasium environment

We built a custom environment (`2048-v0`) that follows Gymnasium conventions. It has a `Discrete(4)` action space (up/down/left/right) and a 4×4 grid for observations. In `step(action)`, we rotate the board so every move can reuse the same “merge left” logic (`_merge_left(...)`), then rotate it back. If the board changes, we add a random tile. The episode ends when `_is_game_over()` detects no empty cells and no valid merges.

### 2.2 Action masking (valid-move detection)

In 2048, lots of moves don't actually do anything. To avoid wasting time on invalid actions and to help learning converge, we added action masking:

- `_get_action_mask()` simulates each action and marks it valid only if it changes the board.
- `reset()` returns `info={"action_mask": mask}` and `step()` includes it in every `info` dict.

The agent uses this mask to zero out Q-values for invalid actions (technically sets them to a huge negative number).

### 2.3 Reward design (reward shaping)

The base reward is whatever score you get from merging tiles. Then we added extra terms in `step()`:

- **Penalty** when the board is getting too full (few empty cells).
- **Bonus** if the biggest tile is in a corner (a standard 2048 trick).

This combination gives the agent more frequent feedback instead of just "you won" or "you lost" at the end.

### 2.4 DQN agent with target network + replay buffer

In `DQNAgent` we implemented the core ideas of DQN: we store experiences as `(state, action, reward, next_state, done)` tuples in a `ReplayBuffer` and sample random batches for training. We keep two networks (`q_net` / `target_net`) —the target network gets updated every so often to stabilize learning. Exploration uses epsilon-greedy with decay. Since 2048 tile values grow exponentially, we preprocess with `log2(max(tile,1))` and reshape to `(batch, 1, 4, 4)` before feeding them into `QNetwork` (a small CNN followed by fully connected layers) that outputs 4 Q-values.

### 2.5 Vectorized training loop

To speed up learning, we run multiple environments in parallel using `gym.vector.SyncVectorEnv`. The training loop handles batched observations and action masks, asks the agent for a batch of actions, and stores all the transitions at once. After we've collected enough data, we call `agent.update()` multiple times to do learning updates.

This is way faster than running one environment at a time.

### 2.6 Testing and baseline

We included a couple evaluation modes:

- **Trained agent test**: loads `dqn_2048.pth` and runs episodes with rendering.
- **Random baseline**: uses `RandomAgent` to show the gap between a learned policy and random moves.

### 2.7 Challenges and future improvements

Some hyperparameters ended up being different between the documentation and actual code (like replay buffer size, batch size). It'd be better to have a single config file.  Also, `game_2048.py` and `env_2048.py` both have overlapping logic for moving/merging and rendering—we could clean that up by having the environment use a `Game2048` instance instead of duplicating everything.

Other things we could add:

- **Reproducibility**: set seeds for NumPy/random/PyTorch explicitly.
- **Evaluation**: track max-tile distributions and auto-generate comparison plots.

---

## 3. What we learned from Part 3

We learned that OOP in RL isn't just about making classes, it's about making sure you can swap pieces in and out easily. By using Gymnasium’s standard API and a shared BaseAgent interface, we could compare a random baseline with DQN or try different reward functions without major rewrites. We also saw how reward shaping is a double-edged sword: it speeds up learning but can accidentally teach the agent weird behaviors. On the engineering side, things like vectorized environments and action masking made a huge difference in both speed and training stability.