import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

# Import your custom environment and ensure it's registered
from game_env import PlatformerEnv, register_env
register_env()  # Ensure environment is registered

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward: {mean_reward:.2f}")

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(os.path.join(self.save_path, "best_model"))
        return True

if __name__ == '__main__':
    # Create log dir
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and verify environment
    try:
        env = gym.make('CustomPlatformer-v0', render_mode=None)
        print("Successfully created CustomPlatformer-v0 environment")
    except gym.error.Error as e:
        print(f"Failed to create environment: {e}")
        raise
    
    # Vectorize environment
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env, log_dir)
    
    # Configure logger
    new_logger = configure(log_dir, ['stdout', 'tensorboard'])
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.00003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log=log_dir
    )
    
    # Set logger
    model.set_logger(new_logger)
    
    # Create callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_path=log_dir)
    
    print("------------- Start Learning -------------")
    try:
        model.learn(total_timesteps=500000, callback=callback, tb_log_name="PPO")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Save final model
    model.save("../model/PPO_0.00003")
    print("------------- Done Learning -------------")
    
    # Test the trained model
    test_env = gym.make('CustomPlatformer-v0', render_mode='human')
    obs, _ = test_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            obs, _ = test_env.reset()
    
    test_env.close()