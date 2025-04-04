import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys
from settings import *
from level import Level
from player import Player
from ui import UI
from main import Game  # Import Game class

# Register the environment
gym.register(
    id='CustomPlatformer-v0',
    entry_point='game_env:PlatformerEnv',
    max_episode_steps=1000,
)

def extract_cell_positions(csv_file, set_1):
    with open(csv_file, "r") as f:
        lines = f.readlines()
    
    list_1 = []
    
    for row, line in enumerate(lines):
        values = list(map(int, line.strip().split(",")))
        for col, value in enumerate(values):
            if value in set_1:
                list_1.append((col*tile_size, 800 - row*tile_size))
    
    return list_1

import csv

def is_there_ground_below_it(csv_file: str, col_no: int, row_no: int) -> bool:
    with open(csv_file, 'r') as file:
        reader = list(csv.reader(file))  

        
        for i in range(row_no, len(reader)):
            row = reader[i]
            if col_no < len(row):
                try:
                    value = int(row[col_no])
                    if value != -1:
                        return 1  
                except ValueError:
                    continue  
    return 0  


class PlatformerEnv(gym.Env):
    """Custom Gymnasium Environment for Mario-like Platformer"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(PlatformerEnv, self).__init__()
        
        self.render_mode = render_mode
        self.terrain = self._load_terrain(f"../levels/{cur_level}/level_{cur_level}_terrain.csv")
        # Information for Edges Detection
        self.csv_file = f"../levels/{cur_level}/level_{cur_level}_terrain.csv"
        set_1 = {0, 2, 3, 12, 14}
        self.list_1 = extract_cell_positions(self.csv_file, set_1)
        self.list_1 = sorted(self.list_1, key=lambda item: item[0])

        # Define action space (0 = Left, 1 = Right, 2 = Jump, 3 = No action)
        self.action_space = spaces.Discrete(4)

        # Observation space: (player_x, player_y, velocity_x, velocity_y, on_ground, next_obstacle_x, next_obstacle_y, next_obstacle2_x, next_obstacle2_y)
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        # )

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(15, 11, 1), dtype=np.float32)
        })

        # Initialize pygame only once
        if not pygame.get_init():
            pygame.init()
            pygame.mixer.init()  # Initialize the mixer for audio

        # Set up screen if rendering
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None

        self.game = None
        self.player_x = 0
        self.player_y = 0
        self.previous_x = 0
        self.total_reward = 0

    def _load_terrain(self, filepath):
        """Load terrain CSV into 2D numpy array"""
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            return np.array([[int(col) for col in row] for row in reader])

    def reset(self, seed=None, options=None):
        """Reset game state at the start of each episode"""
        super().reset(seed=seed)
        
        # Reset the game state
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        
        self.game = Game(external_screen=self.screen)
        player = self.game.level.player.sprites()[0]
        
        # Reset player physics and state
        player.velocity_x = 0
        player.velocity_y = 0
        player.previous_pos = (player.rect.x, player.rect.y)
        player.on_ground = False
        player.on_left = False
        player.on_right = False
        player.on_ceiling = False
        
        # Force initial collision check
        self.game.level.vertical_movement_collision()
        self.game.level.horizontal_movement_collision()
        
        # Get initial player position
        player_position = self.game.level.get_position()
        self.player_x = player_position[0]
        self.player_y = player_position[1]
        
        # Reset tracking variables
        self.previous_x = self.player_x
        self.total_reward = 0
        

        # Calculate nearest obstacles
        observation = self._get_obs()
        
        return observation, {}


    def _get_obs(self):
        """Helper method to get current observation"""
        player_position = self.game.level.get_position()
        player_state = self.game.level.get_player_state()
        collision_info = self.game.level.check_on_ground()
        ################################### SECOND APPROACH #####################################
        grid_x = int(player_position[0] / tile_size)
        grid_y = len(self.terrain) - 1 - int(player_position[1] / tile_size)

        obs_grid = np.zeros((15, 11), dtype=np.float32)

        for dy in range(-5, 10):
            for dx in range(-5, 6):
                level_x = grid_x + dx
                level_y = grid_y + dy
                
                if 0 <= level_x < len(self.terrain[0]) and 0 <= level_y < len(self.terrain):
                    if self.terrain[level_y][level_x] != -1:  # Platform exists
                        obs_grid[dy+5, dx+5] = 1.0  # Normalized to 1.0

        # # Calculate relative positions of obstacles
        # result1 = [(a - player_position[0], b - player_position[1]) for a, b in self.list_1]
        
        # # Find two nearest obstacles to the right
        # nearest_1 = (0, 0)
        # nearest_2 = (0, 0)
        # dist_1 = float('inf')
        # dist_2 = float('inf')

        # for (dx, dy), (orig_x, orig_y) in zip(result1, self.list_1):
        #     if orig_x > player_position[0]:  # Only consider obstacles to the right
        #         distance = dx**2 + dy**2
        #         if distance < dist_1:
        #             nearest_2 = nearest_1
        #             dist_2 = dist_1
        #             nearest_1 = (dx, orig_y - player_position[1])
        #             dist_1 = distance
        #         elif distance < dist_2:
        #             nearest_2 = (dx, orig_y - player_position[1])
        #             dist_2 = distance

        # is_there_ground_or_not = 1
        # # Is there a ground or not below it or not
        # round_off_x = int(player_position[0]/64)
        # round_off_y = int(player_position[1]/64)

        # is_there_ground_or_not = is_there_ground_below_it(self.csv_file,round_off_x,round_off_y)


        print(f"{player_position[0]} {player_position[1]} {player_position[0]-self.previous_x} {player_state['velocity'][1]}")
        # print(obs_grid)
        # return np.array([
        #     player_position[0], player_position[1],           # Position (x, y)
        #     player_position[0]-self.previous_x, player_state['velocity'][1],  # Velocity (vx, vy)
        #     int(collision_info),
        #     is_there_ground_or_not,                              # Grounded status
        #     nearest_1[0], nearest_1[1],                      # Next obstacle 1
        #     nearest_2[0], nearest_2[1]                       # Next obstacle 2
        # ], dtype=np.float32)
        obs_grid = np.expand_dims(obs_grid, axis=-1)  # shape (15, 11, 1)
        obs_vector = np.array([
            player_state['velocity'][0] / 10.0,
            player_state['velocity'][1] / 15.0,
            float(self.game.level.check_on_ground())
        ], dtype=np.float32)

        return {
            "grid": obs_grid,
        }

    def step(self, action):
        """Apply action and update game state"""
        previous_x = self.player_x
        
        # Apply action
        self.game.level.player.sprites()[0].get_input(action)
        
        # Run game logic
        self.game.run()
        
        # Get updated state
        player_position = self.game.level.get_position()
        self.player_x = player_position[0]
        self.player_y = player_position[1]
        positions = self.game.level.get_position_of_start_and_goal()
        
        # Calculate reward and done flag
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Progress reward
        progress = (self.player_x - previous_x) / 10.0
        reward += progress
        
        # 2. Small penalty for existing
        reward -= 0.01
        
        # 3. Jump reward/punishment
        if action == 2:  # Jump action
            collision_info = self.game.level.check_on_ground()
            if collision_info:
                reward += 0.1  # Reward well-timed jumps
            else:
                reward -= 0.2  # Punish spamming jump
        
        # 4. Falling penalty
        player_state = self.game.level.get_player_state()
        if not self.game.level.check_on_ground() and player_state['velocity'][1] < 0:
            reward -= 0.05
        
        # 5. Big penalty for falling in water
        if self.player_y > 700:
            reward -= 20
            print("Fell Into Water!!")
            terminated = True
        
        # 6. Reward for completing level
        if self.player_x >= positions["goal"][0]:
            reward += 100
            print("Level Completed !!!")
            terminated = True
        
        self.total_reward += reward
        self.previous_x = self.player_x
        # Get observation
        observation = self._get_obs()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, {}

    def render_observation(self, obs=None):
        """Debug visualization of the agent's observation"""
        if obs is None:
            obs = self._get_obs()
        
        grid = obs[:165].reshape(15, 11)
        player_state = obs[165:]
        
        print("\n=== Agent's Observation ===")
        
        # Print grid with player at center
        for i, row in enumerate(grid):
            row_str = []
            for j, val in enumerate(row):
                if i == 5 and j == 5:  # Player position
                    row_str.append("P")
                else:
                    row_str.append("█" if val > 0.5 else ".")
            print(" ".join(row_str))
        
        # Print player state
        print(f"\nPlayer State:")
        print(f"X Velocity: {player_state[0]*10:.1f} (normalized: {player_state[0]:.2f})")
        print(f"Y Velocity: {player_state[1]*15:.1f} (normalized: {player_state[1]:.2f})")
        print(f"Grounded: {'YES' if player_state[2] > 0.5 else 'NO'}")
        print("="*30)

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self.screen.fill('grey')
            self.game.run()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.quit()


# At the bottom of game_env.py (after the PlatformerEnv class definition)
def register_env():
    """Register the custom environment"""
    if 'CustomPlatformer-v0' not in gym.envs.registry:
        gym.register(
            id='CustomPlatformer-v0',
            entry_point='game_env:PlatformerEnv',
            max_episode_steps=1000,
        )

# Register the environment when the module is imported
register_env()