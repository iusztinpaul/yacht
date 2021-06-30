import logging

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from yacht.data.renderers import RewardsRenderer


logger = logging.getLogger(__file__)


class RewardsRenderCallback(BaseCallback):
    def __init__(self, storage_path: str, verbose: int = 0):
        super().__init__(verbose)

        self.storage_path = storage_path
        self.all_rewards = []

    def _init_callback(self) -> None:
        self.all_rewards = []

    def _on_rollout_end(self) -> None:
        assert self.model.rollout_buffer.rewards.shape[1] == 1

        rewards = self.model.rollout_buffer.rewards.tolist()
        rewards = [r[0] for r in rewards]
        self.all_rewards.extend(rewards)

    def _on_training_end(self) -> None:
        rewards = pd.DataFrame(data={'Rewards': self.all_rewards})
        renderer = RewardsRenderer(data=rewards)
        renderer.render()
        renderer.save(self.storage_path)

    def _on_step(self) -> bool:
        return True


class LoggerCallback(BaseCallback):
    def __init__(self, collecting_n_steps: int, verbose: int = 0):
        super().__init__(verbose)

        self.collecting_n_steps = collecting_n_steps

    def _on_step(self) -> bool:
        current_episode = self.num_timesteps / self.collecting_n_steps
        if current_episode == int(current_episode):
            logger.info(f'Started episode: {current_episode}.')

        return True
