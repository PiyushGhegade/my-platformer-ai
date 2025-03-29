import pygame
import random
import numpy as np
# from ai_model import AIModel 

class AI:
    def __init__(self):
        pass

    def get_random_number(self):
        return random.randint(0, 1)
    
    # def ai_main(self):
    #     # Initialize AI Model
    #     ai_model = AIModel(state_size=4, action_size=3)  # Example: 4 input features, 3 actions

    #     # Get game state (Example: player_x, player_y, enemy_x, enemy_y)
    #     game_state = [player.x, player.y, enemy.x, enemy.y]

    #     # Normalize the game state
    #     game_state = np.array(game_state) / 500.0  # Normalize for better learning

    #     # AI predicts the best action
    #     action = ai_model.predict(game_state)

    #     # Convert AI output to game action
    #     action_mapping = {0: "left", 1: "right", 2: "jump"}

    #     # Apply AI action in the game
    #     if action_mapping[action] == "left":
    #         player.move_left()
    #     elif action_mapping[action] == "right":
    #         player.move_right()
    #     elif action_mapping[action] == "jump":
    #         player.jump()