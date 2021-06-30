import logging

from stable_baselines3.common.callbacks import BaseCallback


logger = logging.getLogger(__file__)


class LoggerCallback(BaseCallback):
    def __init__(self, collecting_n_steps: int, verbose: int = 0):
        super().__init__(verbose)

        self.collecting_n_steps = collecting_n_steps

    def _on_step(self) -> bool:
        current_episode = self.num_timesteps / self.collecting_n_steps
        if current_episode == int(current_episode):
            logger.info(f'Started episode: {current_episode}.')

        return True
