import gymnasium as gym
import agent_2048
import numpy as np
import time
import argparse
from tqdm import tqdm
import env_2048 # type: ignore
import matplotlib.pyplot as plt

def train(envs, agent, total_steps, plot=False):
    print(f"Starting Vectorized Training for {total_steps} steps...")

    obs, infos = envs.reset()
    pbar = tqdm(range(0, total_steps, envs.num_envs), desc="Training", unit="step")
    
    # Track scores
    episode_rewards = np.zeros(envs.num_envs)
    completed_episode_scores = []

    for step in pbar:
        # A. Handle Action Masks for Batch
        action_masks = None
        if "action_mask" in infos:
             action_masks = np.array(infos["action_mask"])

        # B. Agent acts in batch
        actions = agent.act(obs, action_masks)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        episode_rewards += rewards
        
        # C. Store experiences and handle episode completions
        for i in range(envs.num_envs):
            done = terminated[i] or truncated[i]
            if done:
                completed_episode_scores.append(episode_rewards[i])
                episode_rewards[i] = 0
                
            agent.remember(obs[i], actions[i], rewards[i], next_obs[i], done)

        if len(agent.memory) > agent.batch_size:
            agent.update()

        # D. Move to next state
        obs = next_obs

        # E. Logging
        if step % 1000 == 0:
            avg_max_tile = 2**(np.mean(np.log2(np.max(obs, axis=(1,2)))))
            pbar.set_postfix({
                "Epsilon": f"{agent.epsilon:.2f}", 
                "Avg Max": f"{avg_max_tile:.1f}"
            })

    print("Training finished.")
    
    # Plotting training scores
    if plot and completed_episode_scores:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(completed_episode_scores, alpha=0.3, color='blue', label='Episode Score')
            plt.title('Training Score per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.grid(True)
            # Calculate moving average
            window_size = 100
            scores = np.array(completed_episode_scores)
            plt.plot(scores, alpha=0.3, color='blue', label='Episode Score')
            if len(scores) >= window_size:
                moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(scores)), moving_avg, color='red', label=f'{window_size}-Episode Moving Avg')
            
            plt.legend()
            plt.savefig(f'training_scores_{total_steps}_steps.png')
            print(f"Plot saved to training_scores_{total_steps}_steps.png")
            plt.close()
        except Exception as e:
            print(f"Error creating plot: {e}")

def test(env, agent, episodes=5):
    # Testing loop
    agent.epsilon = 0 # Turn off randomness
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_mask = info.get("action_mask")
            action = agent.act(state, action_mask) # Use single act()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if env.render_mode == 'human':
                time.sleep(0.01)
        
        print(f"Test Episode {episode + 1}, Score: {total_reward:.1f}, Max Tile: {np.max(state)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # TRAIN: Default const = 32M steps
    parser.add_argument('--train', nargs='?', const=32_000_000, default=None, type=int, help='Train agent. Usage: --train (defaults to 32M) or --train 10000')
    
    # TEST: Default const = 5 episodes
    parser.add_argument('--test', nargs='?', const=5, default=None, type=int, help='Test agent. Usage: --test (defaults to 5) or --test 10')
    
    # RANDOM: Default const = 5 episodes
    parser.add_argument('--random', nargs='?', const=5, default=None, type=int, help='Run random agent. Usage: --random (defaults to 5) or --random 10')
    
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no rendering during testing)')
    parser.add_argument('--plot', action='store_true', help='Plot training scores')
    args = parser.parse_args()

    # Logic: If NO specific flags are provided, run the standard Train + Test loop
    run_all = (args.train is None) and (args.test is None) and (args.random is None)

    # --- Random Agent ---
    if args.random is not None:
        print(f"Running Random Agent for {args.random} episodes...")
        env = gym.make('2048-v0', render_mode='human')
        random_agent = agent_2048.RandomAgent(env.action_space)
        test(env, random_agent, episodes=args.random)
        env.close()
    
    # --- Training ---
    if (args.train is not None or run_all) and not (args.random is not None):
        steps = args.train if args.train is not None else 32_000_000
        print(f"Training for {steps} steps...")

        # Create parallel games
        num_envs = 32
        envs = gym.vector.SyncVectorEnv([
            lambda: gym.make('2048-v0', render_mode=None) for _ in range(num_envs)
        ])
        
        agent = agent_2048.DQNAgent(
            grid_size=4, 
            action_space_n=envs.single_action_space.n, 
            epsilon_decay=0.9999
        )
        train(envs, agent, total_steps=steps, plot=args.plot)
        
        agent.save("dqn_2048.pth")
        envs.close()
    
    # --- Testing ---
    if (args.test is not None or run_all) and not (args.random is not None):
        episodes = args.test if args.test is not None else 5
        print(f"Testing for {episodes} episodes...")

        env = gym.make('2048-v0', render_mode='human' if not args.headless else None)
        agent = agent_2048.DQNAgent(grid_size=4, action_space_n=env.action_space.n)
        
        try:
            agent.load("dqn_2048.pth")
            test(env, agent, episodes=episodes)
        except FileNotFoundError:
            print("No model found to test! Train first or ensure dqn_2048.pth exists.")
        
        env.close()