import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from game_env import PlatformerEnv, register_env

register_env()

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
    # Setup
    log_dir = "./ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment with frame stacking
    env = gym.make('CustomPlatformer-v0', render_mode=None)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)  # 4-frame stack
    env = VecMonitor(env, log_dir)
    
    # Hyperparameters optimized for grid observations
    model = PPO(
        "MultiInputPolicy",  # Changed from MlpPolicy!
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger network
        )
    )
    
    # Callback
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_path=log_dir)
    
    # Training
    print("------------- Start Learning -------------")
    try:
        model.learn(
            total_timesteps=1_000_000,  # Increased from 500k
            callback=callback,
            tb_log_name="PPO",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted - saving model...")
        model.save("ppo_interrupted")
    
    # Save and test
    model.save("ppo_platformer")
    print("------------- Training Complete -------------")
    
    # Testing with rendering
    test_env = gym.make('CustomPlatformer-v0', render_mode='human')
    obs = test_env.reset()[0]
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        test_env.render()
        if terminated or truncated:
            obs = test_env.reset()[0]
    test_env.close()