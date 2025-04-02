import gym
from gym import spaces
import numpy as np
import pygame, sys
from settings import *
from level import Level
from player import Player
from ui import UI
from main import Game  # Import Game class


def extract_cell_positions(csv_file, set_1):
    with open(csv_file, "r") as f:
        lines = f.readlines()
    
    list_1 = []
    
    for row, line in enumerate(lines):
        values = list(map(int, line.strip().split(",")))
        for col, value in enumerate(values):
            if value in set_1:
                list_1.append((col*tile_size, 800 - row*tile_size))
            # if value in set_2:
            #     list_2.append((col*tile_size, 800 - row*tile_size))
    
    return list_1

class PlatformerEnv(gym.Env):
    """Custom Gym Environment for Mario-like Platformer"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(PlatformerEnv, self).__init__()

        pygame.init()  # Initialize pygame first
        
        # Set up display BEFORE calling Game
        self.screen = pygame.display.set_mode((screen_width, screen_height))  

        # Now, initialize the game
        self.game = Game(external_screen=self.screen)  # Pass the initialized screen

        #Information for Edges Detection
        csv_file = f"../levels/{cur_level}/level_{cur_level}_terrain.csv"
        set_1 = {0,2, 3, 12,14}
        # set_2 = {2, 3, 14, 15}
        self.list_1 = extract_cell_positions(csv_file, set_1)
        self.list_1 = sorted(self.list_1, key=lambda item: item[0])

        # Define action space (0 = Left, 1 = Right, 2 = Jump, 3 = No action)
        self.action_space = spaces.Discrete(4)

        # Observation space: (player_x, player_y, velocity_y, coins_collected)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32  # ✅ Flattened observation space
        )


        # Get initial player position
        self.reset()

    def reset(self):
        """Reset game state at the start of each episode"""
        # Reset the game state
        self.game = Game(self.screen)
        
        # Get player sprite and reset all tracking variables
        player = self.game.level.player.sprites()[0]
        
        # Reset player physics and state
        player.velocity_x = 0
        player.velocity_y = 0
        player.previous_pos = (player.rect.x, player.rect.y)
        player.on_ground = False  # Will be updated in first collision check
        player.on_left = False
        player.on_right = False
        player.on_ceiling = False
        
        # Force an initial collision check
        self.game.level.vertical_movement_collision()
        self.game.level.horizontal_movement_collision()
        
        # Get initial player position (adjusted for world shift)
        player_position = self.game.level.get_position()
        self.player_x = player_position[0]
        self.player_y = player_position[1]
        
        # Reset tracking variables
        self.previous_x = self.player_x
        self.total_reward = 0
        
        # Calculate nearest obstacle/gap positions
        result1 = [(a - self.player_x, b - self.player_y) for a, b in self.list_1]
        
        # Find the closest obstacle to the right of the player
        closest_obstacle = None
        min_distance = float('inf')
        
        for (dx, dy), (orig_x, orig_y) in zip(result1, self.list_1):
            if orig_x > self.player_x:  # Only consider obstacles to the right
                distance = dx**2 + dy**2  # Squared distance
                if distance < min_distance:
                    min_distance = distance
                    closest_obstacle = (dx, orig_y - self.player_y)
        
        # Default to first obstacle if none found to the right
        if closest_obstacle is None and len(result1) > 0:
            closest_obstacle = (result1[0][0], self.list_1[0][1] - self.player_y)
        
        start_x, start_y = closest_obstacle if closest_obstacle else (0, 0)
        
        # Return observation in Dict format matching observation_space
        return np.array([
            self.player_x, self.player_y,           # Position (x, y)
            player.velocity_x, player.velocity_y,   # Velocity (vx, vy)
            int(player.on_ground),                  # Grounded status (binary)
            start_x, start_y                         # Next obstacle position
        ], dtype=np.float32)

    def step(self, action):
        """Apply action and update game state"""
        previous_x = self.player_x  # Store the previous x-coordinate before the action
        self.game.level.player.sprites()[0].get_input(action)
        # self._apply_action(action)

        self.game.run()
        # Get updated player position
        player_position = self.game.level.get_position()
        positions = self.game.level.get_position_of_start_and_goal()
        self.player_x = player_position[0]
        self.player_y = player_position[1]
        # print(self.player_x)
        result1 = [(a - self.player_x, b - self.player_y) for a, b in self.list_1]
        start_x = result1[1][0]
        start_y = self.list_1[1][1]
        res = float('inf')  # Initialize with a large number

        for i, (a, b) in enumerate(result1):  # Track index using enumerate()
            distance = (a) ** 2 + (b) ** 2  # Compute squared distance
            
            if a > 0 and self.list_1[i][0] > self.player_x and distance < res:
                start_x = a
                start_y = self.list_1[i][1]
                res = distance

        collision_info = self.game.level.check_on_ground()
        player_state = self.game.level.get_player_state()
        #######################################  REWARD SYSYTEM  ####################################
        reward = 0
        done = False

        # 1. Progress reward (scaled by distance to goal)
        progress = (self.player_x - previous_x) / 10.0  # Scale down
        reward += progress
    
        # 2. Small penalty for existing (encourages efficiency)
        reward -= 0.01

        # 4. Reward/punishment for jumping
        if action == 2:  # Jump action
            if collision_info:
                reward += 0.1  # Reward well-timed jumps
            else:
                reward -= 0.2  # Punish spamming jump

        if not collision_info and player_state['velocity'][1] < 0:  # Falling
            reward -= 0.05
        
        # Penalize falling down
        if self.player_y > 700:
            reward -= 20  # Big penalty for falling
            print("Player fell into water! Restarting level...")
            done = True
            # self.reset()  # Reset level (instead of quitting)

        # Reward for completing the level
        elif self.player_x >= positions["goal"][0]:
            reward += 100  # Bonus for completing the level
            print("Level completed!")
            done = True  # Stop episode

        # Update previous_x to the current position
        self.previous_x = self.player_x

        self.total_reward += reward
        
        # Get next obstacle info (from your existing code)
        next_obstacle = [start_x,start_y]

        print(f"{action} {player_state['position'][0]} {player_state['position'][1]} {start_x} {start_y} {player_state['velocity'][0]} {player_state['velocity'][1]} {int(collision_info)} {reward} {self.total_reward} ")
        observation = np.array([
            player_state['position'][0], player_state['position'][1],   # Position (x, y)
            player_state['velocity'][0], player_state['velocity'][1],   # Velocity (vx, vy)
            int(collision_info),                           # Grounded status (binary)
            next_obstacle[0], next_obstacle[1]                          # Next obstacle position
        ], dtype=np.float32)

        return observation, reward, done, {}

    def render(self, mode="human"):
        """Render a single frame of the game"""

        self.screen.fill('grey')  # Ensure background is set
        
        self.game.run()  # Call the main game loop to render objects
        
        pygame.display.update()  # Refresh the screen
        pygame.time.delay(60)  # Small delay for rendering

    def close(self):
        """Close the environment"""
        pygame.quit()



