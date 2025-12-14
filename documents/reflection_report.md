# OOP Group Project – Part 3 Reflection Report (2048 RL)

**Group ID:** 23

**Members:** 陳展皝、簡才斌、鄭祐丞

## Overview

Part 3 is a “Choose Your Own Adventure” task. Our team designed and implemented a **custom 2048 game environment** and trained a **Deep Q-Network (DQN) agent** to play it. The implementation follows the **Gymnasium Env interface** (`reset`, `step`, `render`) and separates responsibilities into agent logic, environment dynamics, and training/testing scripts.

---

## 1. OOP concept(s) implemented in Part 3

### 1.1 Abstraction (Interfaces / Contracts)

A key design goal was to define a clean and minimal contract between components.

- **Agent interface (`BaseAgent`)** (`part3/agent_2048.py`): `BaseAgent` is an abstract base class that requires every agent to implement `act(...)`, so training/testing code can call the same method regardless of the underlying strategy.
- **Environment interface (Gymnasium `gym.Env`)** (`part3/env_2048.py`): `TwentyFortyEightEnv` follows the Gymnasium API (`reset`, `step`, `render`, `close`), which lets us plug the environment into `gym.make(...)` and `gym.vector.SyncVectorEnv(...)` directly.

**Benefit:** the training loop can treat the environment and agents as black boxes with guaranteed methods and return formats.

### 1.2 Inheritance (Reuse + Specialization)

We used inheritance primarily to reuse common behavior and specialize it cleanly.

`RandomAgent` and `DQNAgent` both inherit from `BaseAgent`, so they share the same agent contract but specialize the decision policy (random vs. learned). `DQNAgent` also provides `save/load` to persist a trained model. In addition, `QNetwork` inherits from `nn.Module` and `TwentyFortyEightEnv` inherits from `gym.Env`, which gives us framework-level reuse (parameter management, device movement, standard environment signatures, etc.).

**Benefit:** we get framework integration and code reuse while keeping our project code focused on “2048 + RL” specifics.

### 1.3 Polymorphism (Interchangeable Components)

Polymorphism appears when different concrete types are used through the same interface.

In `part3/main_2048.py`, the same `test(...)` function can evaluate either a `DQNAgent` or a `RandomAgent` because both expose `act(...)`. We also made `act(...)` accept an optional `action_mask`, so the environment can constrain actions without the training loop needing to encode 2048 rules.

**Benefit:** swapping “baseline vs learned agent” requires almost no code changes, improving experimentation speed.

### 1.4 Encapsulation (State + Behavior bundled together)

Each class encapsulates its own internal data and provides methods that control how that data changes.

`TwentyFortyEightEnv` encapsulates the board (`self.board`), reward shaping, termination checks, and rendering state; external code only interacts through `reset/step/render`. Similarly, `ReplayBuffer` hides transition storage, cyclic overwrite logic, and random sampling behind `push(...)` and `sample(...)`.

**Benefit:** fewer accidental side effects; easier debugging because each component has a clear responsibility boundary.

### 1.5 Composition & Separation of Concerns (Modular design)

Part 3 is structured into separate modules:

- `env_2048.py`: environment dynamics + reward + action mask + rendering
- `agent_2048.py`: DQN agent + replay buffer + neural network
- `main_2048.py`: training/testing orchestration

**Benefit:** we can modify the reward function without touching the agent network, or swap a new agent algorithm without rewriting the environment.

---

## 2. Implementation of Part 3

### 2.1 Custom 2048 Gymnasium environment

We implemented a custom environment (`2048-v0`) that follows Gymnasium conventions. The environment defines a `Discrete(4)` action space for the four directions and a `(4,4)` grid observation space storing tile values. In `step(action)`, we rotate the board so all moves reuse the same “merge left” logic (`_merge_left(...)`), then rotate back; if the board changes, we add a random tile. The episode terminates when `_is_game_over()` detects no empty cells and no possible merges.

### 2.2 Action masking (valid-move detection)

2048 has many states where some moves do nothing. To prevent wasting steps and to stabilize learning, the environment provides an **action mask**:

- `_get_action_mask()` simulates each action and marks it valid only if it changes the board.
- `reset()` returns `info={"action_mask": mask}` and `step()` continues to include it in `info`.

The agent consumes this mask and avoids choosing invalid actions by setting their Q-values to a large negative number.

### 2.3 Reward design (reward shaping)

The base reward is the **merge score** (sum of merged tile values during the move). We then added shaping terms in `step()`:

- **Penalty** when empty cells are low (board is “cluttered”).
- **Bonus** if the maximum tile stays in a corner (a common 2048 strategy).

This combination gives the agent a denser learning signal than the sparse “win/lose” outcome.

### 2.4 DQN agent with target network + replay buffer

In `DQNAgent` we implemented the core ideas of DQN: experience replay (store `(state, action, reward, next_state, done)` in `ReplayBuffer` and sample randomly), an online network plus a periodically updated target network (`q_net` / `target_net`) to stabilize bootstrapping, and epsilon-greedy exploration with decay. Because 2048 tile values grow exponentially, we preprocess observations with `log2(max(tile,1))` and reshape them to `(batch, 1, 4, 4)` before feeding them into `QNetwork` (a small CNN + fully connected head) to output 4 action Q-values.

### 2.5 Vectorized training loop

To accelerate learning, we employ `gym.vector.SyncVectorEnv` to execute multiple environments in parallel. The training loop operates on batched observations and corresponding action masks, queries the agent for a batch of actions, and stores the resulting batched transitions in replay memory. After sufficient experience has been accumulated, `agent.update()` is invoked repeatedly to perform learning updates.

This design increases data throughput and improves training efficiency compared to a single-environment loop.

### 2.6 Testing and baseline

We provide multiple evaluation modes:

- **Trained agent test**: loads `dqn_2048.pth` and runs episodes with rendering.
- **Random baseline**: uses `RandomAgent` to demonstrate the difference between learned policy and random play.

### 2.7 Challenges and future improvements

Some hyperparameters/architecture details can drift between documentation and code (e.g., replay size, batch size), so a good improvement would be to centralize configuration. Also, `game_2048.py` and `env_2048.py` both contain movement/merge logic and rendering; we could refactor so the environment composes a `Game2048` instance to reduce duplication.

- **Reproducibility**: set explicit seeds for NumPy/random/PyTorch.
- **Evaluation**: log max-tile distributions and auto-generate plots for comparisons.

---

## 3. What we learned from Part 3

We learned that applying OOP principles to an RL project is not merely about introducing classes, but about keeping components replaceable and system boundaries explicit. By relying on Gymnasium’s standard API and a shared BaseAgent interface, we significantly reduced coupling, which made experimentation such as comparing a random baseline with DQN or testing different reward shaping strategies much easier. We also observed important trade-offs in reward shaping: while it can accelerate learning, it may unintentionally alter the behavior the agent ultimately learns. From an engineering perspective, performance-oriented choices such as vectorized environments and action masking substantially improved both training throughput and stability.