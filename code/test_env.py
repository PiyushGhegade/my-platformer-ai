import gymnasium as gym
from game_env import PlatformerEnv

# Initialize the environment
env = PlatformerEnv(render_mode="human")  # or None if you don't want rendering

# Reset environment (returns observation and info)
obs, info = env.reset()
done = False

# Run for a few steps
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    # Now unpack 5 values from step()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render()  # Render the frame
    
    # Check if episode is done (either terminated or truncated)
    if terminated or truncated:
        obs, info = env.reset()  # Proper reset when episode ends

# Close the environment
env.close()