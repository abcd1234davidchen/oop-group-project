import gymnasium as gym
import env_2048
import agent_2048
import numpy as np
import time
import argparse
from tqdm import tqdm
import torch

# --- 1. THE NEW VECTORIZED TRAINING LOOP (FAST) ---
def train_vectorized(envs, agent, total_steps=1000000000000):
    print(f"Starting Vectorized Training for {total_steps} steps...")
    
    # Reset returns a BATCH of 32 states: (32, 4, 4)
    obs, infos = envs.reset()
    
    # We track total steps, not episodes
    pbar = tqdm(range(0, total_steps, envs.num_envs), desc="Training", unit="step")
    
    for step in pbar:
        # A. Handle Action Masks for Batch
        action_masks = None
        if isinstance(infos, dict) and "action_mask" in infos:
             action_masks = np.array(infos["action_mask"])
        elif isinstance(infos, tuple): 
             # Handle tuple info format if gym version differs
             pass

        # B. Act on Batch (Calls the new act_batch method)
        actions = agent.act_batch(obs, action_masks)
        
        # C. Step all 32 games at once
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        
        # D. Store 32 transitions in memory
        for i in range(envs.num_envs):
            done = terminated[i] or truncated[i]
            agent.remember(obs[i], actions[i], rewards[i], next_obs[i], done)
            
        # E. Train ONCE per batch step (Every 32 game moves)
        if len(agent.memory) > agent.batch_size:
            agent.update()

        obs = next_obs

        # Logging
        if step % 1000 == 0:
            avg_max_tile = np.mean(np.max(obs, axis=(1,2)))
            pbar.set_postfix({
                "Epsilon": f"{agent.epsilon:.2f}", 
                "Avg Max": f"{avg_max_tile:.1f}"
            })

    print("Training finished.")

# --- 2. THE STANDARD TESTING LOOP (Single Env) ---
def test(env, agent, episodes=5):
    print(f"Starting testing for {episodes} episodes...")
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
                time.sleep(0.05)
        
        print(f"Test Episode {episode + 1}, Score: {total_reward:.1f}, Max Tile: {np.max(state)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='mix', choices=['train', 'test', 'mix', 'random'])
    parser.add_argument('--total_steps', type=int, default=12000000, help='Total training steps')
    parser.add_argument('--test_episodes', type=int, default=5)
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()

    # --- 3. TRAIN BLOCK (VECTORIZED) ---
    if args.action in ['train', 'mix']:
        # Create 64 parallel games
        num_envs = 64
        envs = gym.vector.SyncVectorEnv([
            lambda: gym.make('2048-v0', render_mode=None) for _ in range(num_envs)
        ])
        
        # Initialize Agent
        agent = agent_2048.DQNAgent(
            grid_size=4, 
            action_space_n=envs.single_action_space.n, 
            epsilon_decay=0.9999
        )
        
        # Run the FAST loop
        train_vectorized(envs, agent, total_steps=args.total_steps)
        
        agent.save("dqn_2048.pth")
        envs.close()
    
    if args.action in ['test', 'mix']:
        env = gym.make('2048-v0', render_mode='human')
        agent = agent_2048.DQNAgent(grid_size=4, action_space_n=env.action_space.n)
        
        try:
            agent.load("dqn_2048.pth")
            test(env, agent, episodes=args.test_episodes)
        except FileNotFoundError:
            print("No model found to test! Train first.")
        
        env.close()

    if args.action == 'random':
        env = gym.make('2048-v0', render_mode='human')
        random_agent = agent_2048.RandomAgent(env.action_space)
        test(env, random_agent, episodes=args.test_episodes)
        env.close()