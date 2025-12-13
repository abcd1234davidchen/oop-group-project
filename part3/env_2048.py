import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pygame # Re-added for graphics

register(
    id='2048-v0',
    entry_point='env_2048:TwentyFortyEightEnv',
)

class TwentyFortyEightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, size=4):
        self.size = size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 65536, (size, size), dtype=np.int32)
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        
        # --- Rendering Setup (From your original code) ---
        self.window_surface = None
        self.clock = None
        self.cell_size = 100
        self.padding = 10
        self.bg_color = (187, 173, 160)
        self.empty_cell_color = (205, 193, 180)
        self.colors = {
            2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
            16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59),
            128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80),
            1024: (237, 197, 63), 2048: (237, 194, 46),
        }
        self.text_color_dark = (119, 110, 101)
        self.text_color_light = (249, 246, 242)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        self._add_tile()
        self._add_tile()
        
        if self.render_mode == "human":
            self._init_render()
            
        return self.board, {"action_mask": self._get_action_mask()}

    def step(self, action):
        # 1. Fast Numpy Logic
        rotated_board = np.rot90(self.board, k=action)
        new_board, reward = self._merge_left(rotated_board)
        self.board = np.rot90(new_board, k=-action)
        
        if not np.array_equal(rotated_board, new_board):
            self._add_tile()

        # 2. Rewards
        total_reward = float(reward)
        num_empty = len(self.board[self.board == 0])
        if num_empty < 4: total_reward -= 10.0
        if self._is_max_in_corner(): total_reward += 10.0

        terminated = self._is_game_over()
        truncated = False
        
        info = {"max_tile": np.max(self.board), "action_mask": self._get_action_mask()}

        if self.render_mode == "human":
            self.render()

        return self.board, total_reward, terminated, truncated, info

    def _merge_left(self, board):
        new_board = np.zeros_like(board)
        total_score = 0
        for r in range(self.size):
            row = board[r]
            non_zeros = row[row != 0]
            merged = []
            skip = False
            for i in range(len(non_zeros)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(non_zeros) and non_zeros[i] == non_zeros[i+1]:
                    value = non_zeros[i] * 2
                    merged.append(value)
                    total_score += value
                    skip = True
                else:
                    merged.append(non_zeros[i])
            new_board[r, :len(merged)] = merged
        return new_board, total_score

    def _add_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        if len(empty_cells) > 0:
            idx = np.random.choice(len(empty_cells))
            r, c = empty_cells[idx]
            self.board[r, c] = 2 if np.random.random() < 0.9 else 4

    def _is_max_in_corner(self):
        max_val = np.max(self.board)
        corners = [self.board[0,0], self.board[0,3], self.board[3,0], self.board[3,3]]
        return max_val in corners

    def _get_action_mask(self):
        mask = np.zeros(4, dtype=bool)
        for a in range(4):
            rot = np.rot90(self.board, k=a)
            sim, _ = self._merge_left(rot)
            if not np.array_equal(rot, sim):
                mask[a] = True
        return mask

    def _is_game_over(self):
        if np.any(self.board == 0): return False
        for r in range(4):
            for c in range(3):
                if self.board[r, c] == self.board[r, c+1]: return False
        for c in range(4):
            for r in range(3):
                if self.board[r, c] == self.board[r+1, c]: return False
        return True

    # --- GRAPHICS LOGIC RESTORED ---
    def _init_render(self):
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            self.clock = pygame.time.Clock()
            width = self.size * self.cell_size + (self.size + 1) * self.padding
            self.window_surface = pygame.display.set_mode((width, width))
            pygame.display.set_caption("2048 RL")
            self.font = pygame.font.SysFont("Arial", 40, bold=True)

    def render(self):
        if self.render_mode != 'human': return
        if self.window_surface is None: self._init_render()

        self.window_surface.fill(self.bg_color)
        for r in range(self.size):
            for c in range(self.size):
                val = self.board[r, c]
                x = self.padding + c * (self.cell_size + self.padding)
                y = self.padding + r * (self.cell_size + self.padding)
                color = self.colors.get(val, (60, 58, 50))
                if val == 0: color = self.empty_cell_color
                
                pygame.draw.rect(self.window_surface, color, (x, y, self.cell_size, self.cell_size), border_radius=5)
                if val != 0:
                    txt_col = self.text_color_light if val > 4 else self.text_color_dark
                    surf = self.font.render(str(val), True, txt_col)
                    rect = surf.get_rect(center=(x + 50, y + 50))
                    self.window_surface.blit(surf, rect)
        
        pygame.display.update()
        self.clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.close()

    def close(self):
        if self.window_surface is not None:
            pygame.quit()
            self.window_surface = None