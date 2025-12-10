import random
import numpy as np
import pygame
from enum import Enum

class Action(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

class Game2048:
    def __init__(self, size=4, render_mode=None):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        
        # Rendering constants
        self.cell_size = 100
        self.padding = 10
        self.bg_color = (187, 173, 160)
        self.empty_cell_color = (205, 193, 180)
        self.colors = {
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }
        self.text_color_dark = (119, 110, 101)
        self.text_color_light = (249, 246, 242)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self._add_random_tile()
        self._add_random_tile()
        if self.render_mode == 'human':
             self._init_render()
        return self.grid

    def _add_random_tile(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r, c] = 4 if random.random() < 0.1 else 2

    def step(self, action: Action):
        if self.game_over:
            return self.grid, 0, True

        original_grid = self.grid.copy()
        reward = 0
        
        # Rotate grid to standardize movement to "Left"
        # LEFT: 0 rot, UP: 1 rot (90 counter-clockwise so UP becomes LEFT), RIGHT: 2 rot, DOWN: 3 rot
        # Actually easier to think: if I want to move UP, I rotate -90 (90 clockwise) ??
        # Let's verify rotation.
        # If I want to slide LEFT: do nothing.
        # If I want to slide UP: rotate 90 deg counter-clockwise (so top becomes left), slide left, rotate back.
        
        rotations = 0
        if action == Action.UP:
            rotations = 1
        elif action == Action.RIGHT:
            rotations = 2
        elif action == Action.DOWN:
            rotations = 3
            
        self.grid = np.rot90(self.grid, k=rotations)
        
        # Process move left
        self.grid, move_reward = self._move_left(self.grid)
        reward += move_reward
        
        # Rotate back
        self.grid = np.rot90(self.grid, k=-rotations)

        if not np.array_equal(original_grid, self.grid):
            self._add_random_tile()
            if self._check_game_over():
                self.game_over = True
        
        self.score += reward
        return self.grid, reward, self.game_over

    def _move_left(self, grid):
        new_grid = np.zeros_like(grid)
        reward = 0
        for r in range(self.size):
            # 1. Shift non-zeros to left
            row = grid[r]
            shifted = row[row != 0]
            
            # 2. Merge
            merged = []
            skip = False
            for i in range(len(shifted)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(shifted) and shifted[i] == shifted[i+1]:
                    merged_val = shifted[i] * 2
                    merged.append(merged_val)
                    reward += merged_val
                    skip = True
                else:
                    merged.append(shifted[i])
            
            # 3. Fill remaining with zeros
            new_row = np.array(merged + [0] * (self.size - len(merged)), dtype=np.int32)
            new_grid[r] = new_row
            
        return new_grid, reward

    def _check_game_over(self):
        # 1. Check for empty cells
        if 0 in self.grid:
            return False
        
        # 2. Check for possible merges
        for r in range(self.size):
            for c in range(self.size):
                val = self.grid[r, c]
                # Check right
                if c + 1 < self.size and self.grid[r, c+1] == val:
                    return False
                # Check down
                if r + 1 < self.size and self.grid[r+1, c] == val:
                    return False
        return True

    def get_valid_actions(self):
        """Returns a list of valid actions (actions that change the grid)."""
        valid_actions = []
        for action in Action:
            # Create a copy of the grid for simulation
            temp_grid = self.grid.copy()
            
            # Apply the same logic as step, but without random tile or score update
            rotations = 0
            if action == Action.UP:
                rotations = 1
            elif action == Action.RIGHT:
                rotations = 2
            elif action == Action.DOWN:
                rotations = 3
                
            temp_grid = np.rot90(temp_grid, k=rotations)
            temp_grid, _ = self._move_left(temp_grid)
            temp_grid = np.rot90(temp_grid, k=-rotations)
            
            if not np.array_equal(self.grid, temp_grid):
                valid_actions.append(action)
        return valid_actions

    def get_num_empty_cells(self):
        return np.sum(self.grid == 0)

    def is_max_tile_in_corner(self):
        max_val = np.max(self.grid)
        corners = [
            self.grid[0, 0],
            self.grid[0, self.size - 1],
            self.grid[self.size - 1, 0],
            self.grid[self.size - 1, self.size - 1]
        ]
        return max_val in corners

    def get_state(self):
        return self.grid

    def get_score(self):
        return self.score
    
    # --- Rendering ---
    def _init_render(self):
        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            self.clock = pygame.time.Clock()
            
            width = self.size * self.cell_size + (self.size + 1) * self.padding
            height = width
            
            self.window_surface = pygame.display.set_mode((width, height))
            pygame.display.set_caption("2048 RL")
            self.font = pygame.font.SysFont("Arial", 40, bold=True)

    def render(self):
        if self.render_mode != 'human':
            return
            
        if self.window_surface is None:
            self._init_render()

        self.window_surface.fill(self.bg_color)
        
        for r in range(self.size):
            for c in range(self.size):
                val = self.grid[r, c]
                x = self.padding + c * (self.cell_size + self.padding)
                y = self.padding + r * (self.cell_size + self.padding)
                
                # Draw cell background
                color = self.colors.get(val, (60, 58, 50)) # Default for > 2048
                if val == 0:
                    color = self.empty_cell_color
                    
                pygame.draw.rect(
                    self.window_surface, 
                    color, 
                    pygame.Rect(x, y, self.cell_size, self.cell_size),
                    border_radius=5
                )
                
                # Draw text
                if val != 0:
                    text_color = self.text_color_light if val > 4 else self.text_color_dark
                    text_surface = self.font.render(str(val), True, text_color)
                    text_rect = text_surface.get_rect(center=(x + self.cell_size/2, y + self.cell_size/2))
                    self.window_surface.blit(text_surface, text_rect)
        
        pygame.display.update()
        self.clock.tick(30) # Limit FPS
        
        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self.window_surface is not None:
            pygame.quit()
            self.window_surface = None

