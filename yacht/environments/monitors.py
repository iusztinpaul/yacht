from typing import Tuple

import gym
import pandas as pd
from stable_baselines3.common.monitor import Monitor

from yacht import utils
from yacht.data.renderers import RewardsRenderer


class RewardRendererMonitor(Monitor):
    def __init__(
            self,
            env: gym.Env,
            final_step: int,
            storage_dir: str,
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
    ):
        super(RewardRendererMonitor, self).__init__(
            env,
            storage_dir,
            allow_early_resets,
            reset_keywords,
            info_keywords
        )

        self.final_step = final_step
        self.storage_path = utils.build_graphics_path(storage_dir, 'train_episode_rewards.png')

    def step(self, action):
        observation, reward, done, info = super().step(action)

        if self.total_steps == self.final_step:
            rewards = pd.DataFrame(data={'Rewards': self.episode_returns})
            renderer = RewardsRenderer(data=rewards)
            renderer.render()
            renderer.save(self.storage_path)

        return observation, reward, done, info
