# OOP Group Project
This repository contains three implementations of reinforcement learning agents using Object-Oriented Programming (OOP) principles. The project is divided into three parts:

- Part 1: mountain car
- Part 2: frozen lake
- Part 3: custom 2048 game environment with Deep Q-Network (DQN) agent

For the first part, the run is a success. The second part gets 63% score. The third part gets high scores consistently.

## Project Overview
This project all uses OpenAI Gymnasium environments to demonstrate different reinforcement learning techniques.

### Part 1: Mountain Car
This is one of the example scene in OpenAI Gymnasium. The objective is to drive a car up a steep hill using momentum. The agent uses Q-learning with a discretized state space to learn the optimal policy.

### Part 2: Frozen Lake
This part implements a Q-learning agent to navigate a frozen lake environment. The agent learns to avoid holes and reach the goal by updating its Q-values based on the rewards received. We modified the three parameters to improve the performance of the agent, achieving a success rate of 63%.

### Part 3: Custom 2048 Game Environment with DQN Agent
We implemented our own 2048 in a custom OpenAI Gymnasium environment. A DQN agent is designed and trained to play the game using experience replay and a target network. The agent can achieve high scores consistently after training.

## Requirements and Setup
This project is managed using UV, use `uv sync` to install dependencies and set up the environment if UV is installed.

Alternatively, you can manually install the required packages using pip:
```bash
pip install "gymnasium[classic-control]" numpy pygame torch matplotlib tqdm
```

## Running the Code
To run each part of the project, navigate to the respective directory and execute the main script with the desired options.

### Part 1: mountain car
```bash
uv run main_mountain_car.py [--train] [--render] [--episodes [EPISODES]]
```
- --train: Default to test, if specified, will train the agent.  
- --render: Render the environment during training/testing.  
- --episodes: Number of episodes to train/test the agent (default: 10).
### Part 2: frozen lake
```bash
uv run main_frozen_lake.py [--train]
```
- --train: Default to test and run for 1000 episodes, if specified, will train the agent with 15000 episodes.
### Part 3: custom 2048 game environment with DQN agent
```bash
uv run main_2048.py [--train[TRAIN]] [--test[TEST]] [--random[RANDOM]] [--render [True|False]] [--plot]
```
if no arguments are provided, both training and testing will be executed.
- --train: Train the DQN agent for a specified number of steps (default: 32 million). If test is not specified, training will not run.
- --test: Test the trained DQN agent for a specified number of episodes (default: 5). If train is not specified, testing will not run.
- --random: Run a random agent for a specified number of episodes (default: 5). Training and testing will be skipped if this is specified.
- --render: Render the environment during testing (default: True).
- --plot: Plot the training scores after training (default: False).

## Contribution Table
| Team Member | Contribution |
|-------------|--------------|
|B123245006 | Part 2, Part 3 Main, Part 3 Agent|
|B123245001 | README, UML, reflection report, bug fix|
|B123245003 | Part 3 Environment, Part 3 Training, Slides|
