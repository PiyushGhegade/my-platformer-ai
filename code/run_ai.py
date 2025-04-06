import gymnasium as gym
from stable_baselines3 import PPO
from game_env import PlatformerEnv
import numpy as np
import sys

# Load the trained model
model = PPO.load(f"../model/{sys.argv[1]}")

# Verify the model's observation space
print("Model's observation space:", model.observation_space)

# Create the custom environment
env = PlatformerEnv(render_mode='human')

# Run episodes indefinitely
while True:
    # Reset the environment and get the initial observation
    obs_dict, _ = env.reset()
    
    # Convert observation to match what the model expects
    # The model expects (15, 11, 4) but environment provides (15, 11, 1)
    # We'll replicate the single channel to 4 channels
    original_grid = np.array(obs_dict["grid"], dtype=np.float32)
    expanded_grid = np.repeat(original_grid, 4, axis=-1)  # Now shape (15, 11, 4)
    
    obs = {
        "grid": expanded_grid
    }
    
    done = False

    # Run the episode
    while not done:
        # Predict the action
        action, _ = model.predict(obs, deterministic=True)

        # Take action in the environment
        next_obs_dict, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Prepare next observation (again expanding to 4 channels)
        original_grid = np.array(next_obs_dict["grid"], dtype=np.float32)
        obs = {
            "grid": np.repeat(original_grid, 4, axis=-1)
        }

        # Render the game
        env.render()

    print("Episode completed. Resetting environment...")

# Close environment (never reached due to infinite loop)
env.close()