import logging
import os

import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from yacht import utils

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
        # If total_timesteps < 10 the division + rounding will return 0.
        if self.log_frequency == 0:
            self.log_frequency = 1

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_frequency == 0:
            logger.info(f'Timestep [{self.num_timesteps} / {self.total_timesteps}]')

        return True


class WandBCallback(BaseCallback):
    def __init__(self, storage_dir: str, verbose: int = 0):
        super().__init__(verbose)

        self.storage_dir = storage_dir

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

        if utils.get_experiment_tracker_name(self.storage_dir) == 'wandb':
            best_model_path = utils.build_best_checkpoint_path(self.storage_dir)
            if os.path.exists(best_model_path):
                wandb.save(best_model_path)
