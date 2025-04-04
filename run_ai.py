import gymnasium as gym
from stable_baselines3 import PPO
from game_env import PlatformerEnv
import sys
import numpy as np

# Load trained model
model = PPO.load(f"../model/{sys.argv[1]}")

# Create environment
env = PlatformerEnv(render_mode='human')  # Explicitly set render mode

# Run indefinitely
while True:
    # Reset the environment and get initial observation
    obs, _ = env.reset()  # Note: Gymnasium returns (observation, info)
    
    # Ensure observation is a numpy array
    if isinstance(obs, tuple):
        obs = obs[0]  # Take just the observation part
    obs = np.array(obs, dtype=np.float32)  # Convert to numpy array if not already
    
    # Render immediately after reset
    env.render()

    done = False

    while not done:
        # Predict the next action
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action and get the updated observation
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Ensure observation remains in correct format
        obs = np.array(obs, dtype=np.float32)
        
        # Render the environment
        env.render()

    print("Episode completed. Resetting environment...")
    # Add a small delay if needed
    # import time
    # time.sleep(1)

# Close the environment (unreachable in this loop)
env.close()