import gym
from stable_baselines3 import PPO
from game_env import PlatformerEnv

# Create environment
env = PlatformerEnv()

# Train using Proximal Policy Optimization (PPO)
model = PPO(    
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0001,  # Lower LR if training is unstable
    n_steps=2048,  # Increase if training is too noisy
    batch_size=64,  # Ensure batches aren't too small
    gamma=0.99,  # Keeps long-term rewards important
    )
model.learn(total_timesteps=200)

# Save trained modele
model.save(f"../model/Himanshu")
env.close()
