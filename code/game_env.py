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

import csv


class PlatformerEnv(gym.Env):
    """Custom Gymnasium Environment for Mario-like Platformer"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(PlatformerEnv, self).__init__()
        
        self.render_mode = render_mode
        self.terrain = self._load_map(f"../levels/{cur_level}/level_{cur_level}_terrain.csv")
        self.front_tree = self._load_map(f"../levels/{cur_level}/level_{cur_level}_fg_palms.csv")
        self.crates = self._load_map(f"../levels/{cur_level}/level_{cur_level}_crates.csv")
        self.coins = self._load_map(f"../levels/{cur_level}/level_{cur_level}_coins.csv")
        # Information for Edges Detection
        self.csv_file = f"../levels/{cur_level}/level_{cur_level}_terrain.csv"

        # Define action space (0 = Left, 1 = Right, 2 = Jump, 3 = No action)
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(15, 11, 1), dtype=np.float32),
            "grid_enemy": spaces.Box(low=0, high=1, shape=(15, 11, 1), dtype=np.float32)
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
        self.last_coin_count = 0
        self.last_kill_count = 0
        self.last_health = 0


    def _load_map(self, filepath):
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
        enemies_position = self.game.level.get_enemy_positions()
        ################################### SECOND APPROACH #####################################
        grid_x = int(player_position[0] / tile_size)
        grid_y = len(self.terrain) - 1 - int(player_position[1] / tile_size)

        obs_grid = np.zeros((15, 11), dtype=np.float32)
        obs_grid_enemy = np.zeros((15, 11), dtype=np.float32)



        for dy in range(-5, 10):
            for dx in range(-5, 6):
                level_x = grid_x + dx
                level_y = grid_y + dy
                
                if 0 <= level_x < len(self.terrain[0]) and 0 <= level_y < len(self.terrain):
                    if self.terrain[level_y][level_x] != -1:  # Platform exists
                        obs_grid[dy+5, dx+5] = 1.0  # Normalized to 1.0
                    if self.front_tree[level_y][level_x] != -1:
                        obs_grid[dy+5, dx+5] = 1.0
                    if self.crates[level_y][level_x] != -1:
                        obs_grid[dy+5, dx+5] = 1.0
                    if self.coins[level_y][level_x] == 1:
                        obs_grid[dy+5, dx+5] = 2.0
                    if self.coins[level_y][level_x] == 0:
                        obs_grid[dy+5, dx+5] = 3.0

                for (enemy_x, enemy_y) in enemies_position:
                    if level_x == enemy_x and level_y == enemy_y:
                        obs_grid_enemy[dy+5, dx+5] = 1.0
                              

        # print(f"{player_position[0]} {player_position[1]} {player_position[0]-self.previous_x} {player_state['velocity'][1]}")
        # print(obs_grid)
        # print(obs_grid_enemy)
        obs_grid = np.expand_dims(obs_grid, axis=-1)  # shape (15, 11, 1)
        obs_grid_enemy = np.expand_dims(obs_grid_enemy, axis=-1)  # shape (15, 11, 1)
        return {
            "grid": obs_grid,
            "grid_enemy": obs_grid_enemy
        }

    def step(self, action):
        previous_x = self.player_x
        
        # Apply action and run game logic
        self.game.level.player.sprites()[0].get_input(action)
        self.game.run()
        
        # Get updated state
        player_position = self.game.level.get_position()
        self.player_x = player_position[0]
        self.player_y = player_position[1]
        positions = self.game.level.get_position_of_start_and_goal()
        
        # Initialize rewards and flags
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Progress reward (normalized for FPS)
        progress = (self.player_x - previous_x)
        if progress > 0:
            reward += progress * 0.05  # Reduced multiplier for high FPS
        
        # 2. Tiny time penalty (scaled for FPS)
        reward -= 0.001  # Much smaller penalty per frame
        
        # 3. Jump mechanics (less frequent rewards)
        if action == 2:  # Jump action
            collision_info = self.game.level.check_on_ground()
            if collision_info:  # Only reward successful landings, not takeoffs
                reward += 0.05  # Smaller jump reward
            # No penalty for jumping in air to encourage exploration
        
        
        # 5. Terminal states (keep impactful)
        if self.player_y > 700:  # Fell in water
            print("Fell in water !!!")
            reward -= 5  # Reduced but still significant
            terminated = True
        
        # 6. Completion reward (keep large but scale progress rewards smaller)
        if self.player_x >= positions["goal"][0]:
            print("Level complete!!!")
            reward += 100  # Kept large but balanced with other rewards
            terminated = True
        
        # 7. Collectibles (accumulated rewards)
        new_coins = self.game.coins - self.last_coin_count
        if new_coins > 0:
            reward += new_coins * 0.1  # Reward per coin collected
            self.last_coin_count = self.game.coins
        
        # 8. Combat (event-based rewards)
        new_kills = self.game.enemy_killed - self.last_kill_count
        if new_kills > 0:
            reward += new_kills * 1.0  # Significant reward per kill
            self.last_kill_count = self.game.enemy_killed
        
        # 9. Health (change-based penalty)
        health_lost = (self.last_health - self.game.cur_health)
        if health_lost > 0:
            reward -= health_lost * 0.5  # Penalty when actually losing health
        self.last_health = self.game.cur_health
        
        # 10. Stuck detection (frame-independent)
        if abs(self.player_x - previous_x) < 0.1 and self.game.level.check_on_ground():
            reward -= 0.01  # Small penalty for being stuck
        
        self.total_reward += reward
        self.previous_x = self.player_x
        
        return self._get_obs(), reward, terminated, truncated, {}


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