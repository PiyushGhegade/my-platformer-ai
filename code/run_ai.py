import gymnasium as gym
from stable_baselines3 import PPO
from game_env import PlatformerEnv
import numpy as np

# Load the trained model
model = PPO.load("ppo_platformer_improved")

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
    original_grid_enemy = np.array(obs_dict["grid_enemy"], dtype=np.float32)
    expanded_grid = np.repeat(original_grid, 4, axis=-1)  # Now shape (15, 11, 4)
    expanded_grid_enemy = np.repeat(original_grid_enemy, 4, axis=-1)  # Now shape (15, 11, 4)
    
    obs = {
        "grid": expanded_grid,
        "grid_enemy": expanded_grid_enemy
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
        original_grid_enemy = np.array(next_obs_dict["grid_enemy"], dtype=np.float32)
        obs = {
            "grid": np.repeat(original_grid, 4, axis=-1),
            "grid_enemy": np.repeat(original_grid_enemy, 4, axis=-1)
        }

        # Render the game
        env.render()

    print("Episode completed. Resetting environment...")

# Close environment (never reached due to infinite loop)
env.close()