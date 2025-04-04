import gym
from game_env import PlatformerEnv

# Initialize the environment
env = PlatformerEnv()

# Reset environment
obs = env.reset()
done = False

# Run for a few steps
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, _ = env.step(action)
    env.render()  # Render the frame
    # if done:
    #     break  # Stop if episode ends

# Close the environment
env.close()
