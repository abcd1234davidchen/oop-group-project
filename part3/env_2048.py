import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import game_2048 as g2048

register(
    id='2048-v0',
    entry_point='env_2048:TwentyFortyEightEnv',
)

class TwentyFortyEightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=4):
        self.size = size
        self.render_mode = render_mode
        self.game = g2048.Game2048(size=size, render_mode=render_mode)
        
        # Action space: 0: Left, 1: Up, 2: Right, 3: Down
        self.action_space = spaces.Discrete(len(g2048.Action))
        
        # Observation space: 4x4 grid. Values are powers of 2.
        # Max tile is theoretically large, but we can set a reasonable upper bound or use high=inf
        # Using 65536 as a safe upper bound for 4x4
        self.observation_space = spaces.Box(
            low=0, 
            high=65536, 
            shape=(size, size), 
            dtype=np.int32
        )

    def _get_action_mask(self):
        valid_actions = self.game.get_valid_actions()
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action in valid_actions:
            mask[action.value] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             # If game supported seeding, we would pass it. 
             # Currently Game2048 uses random module. 
             # We can seed python's random or numpy if we want to support it deeply.
             # For now, just call reset.
             pass
        
        grid = self.game.reset()
        info = {"action_mask": self._get_action_mask()}
        return grid, info

    def step(self, action):
        # Map integer action to Action enum
        try:
            game_action = g2048.Action(action)
        except ValueError:
             # Handle invalid actions if necessary, though gym samples should be valid
             game_action = g2048.Action.LEFT 

        grid, reward, terminated = self.game.step(game_action)
        
        # Add penalty for having too many non-empty blocks (or few empty blocks)
        # Strategy: Penalize based on the number of occupied cells
        # Or: Penalize if number of empty cells is low
        
        num_empty = self.game.get_num_empty_cells()
        # Example penalty: -10 if empty cells < 4
        if num_empty < 4:
            reward -= 10
            
        # Reward for keeping max tile in corner (strategy)
        if self.game.is_max_tile_in_corner():
            reward += 10 # Encourage keeping max tile in corner
        
        # Truncated is used for time limits, which we don't strictly enforce here but could
        truncated = False
        
        info = {
            "score": self.game.get_score(),
            "max_tile": np.max(grid),
            "action_mask": self._get_action_mask()
        }
        
        if self.render_mode == "human":
            self.render()

        return grid, float(reward), terminated, truncated, info

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()
