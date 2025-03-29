import gym
from stable_baselines3 import PPO
from game_env import PlatformerEnv

# Load trained model
model = PPO.load("platformer_ai_level_1")

# Create environment
env = PlatformerEnv()

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()
