import gym
from stable_baselines3 import PPO
from game_env import PlatformerEnv

# Load trained model
model = PPO.load("platformer_ai_level_1")

# Create environment
env = PlatformerEnv()

# Reset the environment and get initial observation
obs = env.reset()

# Render immediately after reset (before agent starts acting)
env.render()

done = False

while not done:
    # Predict the next action
    action, _states = model.predict(obs)
    
    # Take the action and get the updated observation, reward, and done status
    obs, reward, done, _ = env.step(action)
    
    # Render the environment at each step
    env.render()

# Close the environment when done
env.close()
