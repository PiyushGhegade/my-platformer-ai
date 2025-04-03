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

class PlatformerEnv(gym.Env):
    """Custom Gymnasium Environment for Mario-like Platformer"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(PlatformerEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # Information for Edges Detection
        csv_file = f"../levels/{cur_level}/level_{cur_level}_terrain.csv"
        set_1 = {0, 2, 3, 12, 14}
        self.list_1 = extract_cell_positions(csv_file, set_1)
        self.list_1 = sorted(self.list_1, key=lambda item: item[0])

        # Define action space (0 = Left, 1 = Right, 2 = Jump, 3 = No action)
        self.action_space = spaces.Discrete(4)

        # Observation space: (player_x, player_y, velocity_x, velocity_y, on_ground, next_obstacle_x, next_obstacle_y, next_obstacle2_x, next_obstacle2_y)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

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
        
        # Calculate relative positions of obstacles
        result1 = [(a - player_position[0], b - player_position[1]) for a, b in self.list_1]
        
        # Find two nearest obstacles to the right
        nearest_1 = (0, 0)
        nearest_2 = (0, 0)
        dist_1 = float('inf')
        dist_2 = float('inf')

        for (dx, dy), (orig_x, orig_y) in zip(result1, self.list_1):
            if orig_x > player_position[0]:  # Only consider obstacles to the right
                distance = dx**2 + dy**2
                if distance < dist_1:
                    nearest_2 = nearest_1
                    dist_2 = dist_1
                    nearest_1 = (dx, orig_y - player_position[1])
                    dist_1 = distance
                elif distance < dist_2:
                    nearest_2 = (dx, orig_y - player_position[1])
                    dist_2 = distance

        return np.array([
            player_position[0], player_position[1],           # Position (x, y)
            player_state['velocity'][0], player_state['velocity'][1],  # Velocity (vx, vy)
            int(collision_info),                              # Grounded status
            nearest_1[0], nearest_1[1],                      # Next obstacle 1
            nearest_2[0], nearest_2[1]                       # Next obstacle 2
        ], dtype=np.float32)

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
            terminated = True
        
        # 6. Reward for completing level
        if self.player_x >= positions["goal"][0]:
            reward += 100
            terminated = True
        
        self.total_reward += reward
        
        # Get observation
        observation = self._get_obs()
        
        # Render if needed
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, {}

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