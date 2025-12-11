import gymnasium as gym
import env_2048
import agent_2048
import numpy as np
import time
import argparse
import tqdm

def train(env, agent, episodes=100):
    print(f"Starting training for {episodes} episodes...")
    pbar = tqdm(range(episodes), desc="Training", unit="ep")
    for episode in pbar:
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_mask = info.get("action_mask")
            action = agent.act(state, action_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            total_reward += reward
            
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Max Tile: {np.max(state)}")
    
    print("Training finished.")

def test(env, agent, episodes=5):
    print(f"Starting testing for {episodes} episodes...")
    # Turn off exploration
    agent.epsilon = 0
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action_mask = info.get("action_mask")
            action = agent.act(state, action_mask)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Slow down for visualization if rendering
            if env.render_mode == 'human':
                time.sleep(0.1)
        
        print(f"Test Episode {episode + 1}, Score: {total_reward}, Max Tile: {np.max(state)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='mix', choices=['train', 'test', 'mix'], help='Action: train, test, or mix')
    parser.add_argument('--train_episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--test_episodes', type=int, default=5, help='Number of testing episodes')
    args = parser.parse_args()
    # 1. Train
    env = gym.make('2048-v0', render_mode=None)
    agent = agent_2048.DQNAgent(grid_size=4, action_space_n=env.action_space.n, epsilon_decay=0.999)
    
    train(env, agent, episodes=args.train_episodes) # Short training for demonstration
    
    # Save agent
    agent.save("dqn_2048.pth")
    
    # 2. Test with rendering
    env.close()
    env = gym.make('2048-v0', render_mode='human')
    test(env, agent, episodes=1)
    env.close()
