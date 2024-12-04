from gymnasium import RewardWrapper
import numpy as np


class CustomRewardWrapper(RewardWrapper):
    def __init__(self, env, action_change_threshold=0.1):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)

        # Add a reward
        reward = reward

        return obs, reward, done, _, info