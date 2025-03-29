import gym
from gym import spaces
import numpy as np
import pygame, sys
from settings import *
from level import Level
from player import Player
from ui import UI
from main import Game  # Import Game class


def extract_cell_positions(csv_file, set_1, set_2):
    with open(csv_file, "r") as f:
        lines = f.readlines()
    
    list_1 = []
    list_2 = []
    
    for row, line in enumerate(lines):
        values = list(map(int, line.strip().split(",")))
        for col, value in enumerate(values):
            if value in set_1:
                list_1.append((col*tile_size, 800 - row*tile_size))
            if value in set_2:
                list_2.append((col*tile_size, 800 - row*tile_size))
    
    return list_1, list_2

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
        csv_file = "../levels/2/level_2_terrain.csv"
        set_1 = {0, 3, 12, 15}
        set_2 = {2, 3, 14, 15}
        self.list_1, self.list_2 = extract_cell_positions(csv_file, set_1, set_2)
       

        # Define action space (0 = Left, 1 = Right, 2 = Jump, 3 = No action)
        self.action_space = spaces.Discrete(4)

        # Observation space: (player_x, player_y, velocity_y, coins_collected)
        self.observation_space = spaces.Box(
        low=np.array([0, 0, 0, 0, 0, 0]), 
        high=np.array([4000, 800, 4000, 800 , 4000, 800]), 
        dtype=np.float32
        )

        # Get initial player position
        self.reset()

    def reset(self):
        """Reset game state at the start of each episode"""
        self.game = Game(self.screen)
        print(self.list_1)
        # Get player position
        player_position = self.game.level.get_position()
        self.player_x = player_position[0]
        self.player_y = player_position[1]
        self.velocity_y = 0
        self.previous_x = self.player_x  # Store the initial position for comparison
        result1 = [(a - self.player_x, b - self.player_y) for a, b in self.list_1]
        result2 = [(a - self.player_x, b - self.player_y) for a, b in self.list_2]
        start_x = self.player_x - result1[0][0]
        start_y = self.player_x - result1[0][1]
        end_x = self.player_x - result2[0][0]
        end_y = self.player_x - result2[0][1]

        self.total_reward = 0

        return np.array([self.player_x, self.player_y, start_x, start_y, end_x, end_y], dtype=np.float32)

    def _apply_action(self, action):
        """Simulate key presses for the agent"""
        keys = {
            0: pygame.K_LEFT,
            1: pygame.K_RIGHT,
            2: pygame.K_SPACE,  # Jump
            3: None,  # No action
        }

        if action in keys and keys[action]:
            if keys[action] is not None:
                # Simulate key press
                pygame.key.set_repeat(1, 1)  # Set a key repeat rate for held keys
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=keys[action]))
                pygame.event.post(pygame.event.Event(pygame.KEYUP, key=keys[action]))  # Release 

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
        result2 = [(a - self.player_x, b - self.player_y) for a, b in self.list_2]
        start_x = self.player_x - result1[0][0]
        start_y = self.player_x - result1[0][1]
        end_x = self.player_x - result2[0][0]
        end_y = self.player_x - result2[0][1]
        res = float('inf')
        for a, b in result1:
            if a > 0 and b > 0 and ((self.player_x - a) ** 2 + (self.player_y - b) ** 2) < res:
                start_x = a
                start_y = b
                res = (self.player_x - a) ** 2 + (self.player_y - b) ** 2
        res = float('inf')
        for a, b in result2:
            if a > 0 and b > 0 and ((self.player_x - a) ** 2 + (self.player_y - b) ** 2) < res:
                end_x = a
                end_y = b
                res = (self.player_x - a) ** 2 + (self.player_y - b) ** 2

        


        # print(self.player_x, positions["goal"][0], positions["player_start"][0])
        # Reward system
        reward = 0
        done = False
        # Reward for collecting coins
        # reward += self.game.coins * 10  

        # movement_reward = (self.player_x - previous_x) * 0.0001  
        # reward += movement_reward  # Encourage moving right

        x0 =  positions["goal"][0]-positions["player_start"][0]
        x = positions["goal"][0]-self.player_x
        progress_towards_goal = (x0 - x) / x0
        reward += progress_towards_goal * 0.01
        # # Penalize moving left
        # if self.player_x < previous_x:
        #     reward -= 0.05

        # Penalize staying in one place
        if self.player_x == previous_x:
            reward -= 0.001 

        # Penalize jumping (action == 2)
        if action == 2:  # If the action is Jump
            reward -= 0.005  # Small penalty for jumping

        # Penalize falling down
        if self.player_y > 700:
            reward -= 20  # Big penalty for falling
            print("Player fell into water! Restarting level...")
            done = True
            self.reset()  # Reset level (instead of quitting)

        # Reward for completing the level
        elif self.player_x >= positions["goal"][0]:
            reward += 300  # Bonus for completing the level
            print("Level completed!")
            done = True  # Stop episode

        # Update previous_x to the current position
        self.previous_x = self.player_x

        self.total_reward += reward

        # Print the reward for the current step
        print(f"{action} {self.player_x} {reward} {self.total_reward} {start_x} {start_y} {end_x} {end_y}")

        return np.array([self.player_x, self.player_y, start_x, start_y, end_x, end_y], dtype=np.float32), self.total_reward, done, {}

    def render(self, mode="human"):
        """Render a single frame of the game"""

        self.screen.fill('grey')  # Ensure background is set
        
        self.game.run()  # Call the main game loop to render objects
        
        pygame.display.update()  # Refresh the screen
        pygame.time.delay(60)  # Small delay for rendering

    def close(self):
        """Close the environment"""
        pygame.quit()



