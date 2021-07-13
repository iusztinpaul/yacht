import logging

import wandb
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__file__)


class LoggerCallback(BaseCallback):
    def __init__(
            self,
            collect_n_times: int,
            collecting_n_steps: int,
            verbose: int = 0
    ):
        super().__init__(verbose)

        self.collect_n_times = collect_n_times
        self.collecting_n_steps = collecting_n_steps

    def _on_step(self) -> bool:
        current_episode = self.num_timesteps / self.collecting_n_steps
        if current_episode == int(current_episode):
            logger.info(f'Finished training for collect_n_time: [{current_episode}/{self.collect_n_times}].')

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
