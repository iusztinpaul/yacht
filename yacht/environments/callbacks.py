import logging
from math import log10

import wandb
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__file__)


class LoggerCallback(BaseCallback):
    def __init__(
            self,
            total_timesteps: int,
            verbose: int = 0
    ):
        super().__init__(verbose)

        self.total_timesteps = total_timesteps
        self.log_frequency = total_timesteps // 10

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_frequency == 0:
            logger.info(f'Timestep [{self.num_timesteps} / {self.total_timesteps}]')

        return True


class WandBCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        policy = self.locals['self'].policy

        wandb.watch(
            (
                policy.features_extractor,
                policy.mlp_extractor,
                policy.value_net,
                policy.action_net
            ),
            log='parameters',
            log_freq=100
        )

    def _on_training_end(self) -> None:
        policy = self.locals['self'].policy

        wandb.unwatch(
            (
                policy.features_extractor,
                policy.mlp_extractor,
                policy.value_net,
                policy.action_net
            )
        )
