from stable_baselines3.common.callbacks import BaseCallback

from yacht import utils, Mode
from yacht.data.renderers import RewardsRenderer
from yacht.logger import Logger


class LoggerCallback(BaseCallback):
    def __init__(
            self,
            logger: Logger,
            total_timesteps: int,
            verbose: int = 0
    ):
        super().__init__(verbose)

        self.logger = logger
        self.total_timesteps = total_timesteps
        self.log_frequency = total_timesteps // 10
        # If total_timesteps < 10 the division + rounding will return 0.
        if self.log_frequency == 0:
            self.log_frequency = 1

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_frequency == 0:
            self.logger.info(f'Timestep [{self.num_timesteps} / {self.total_timesteps}]')

        return True


class RewardsRenderCallback(BaseCallback):
    def __init__(self, total_timesteps: int, storage_dir: str, mode: Mode, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.renderer = RewardsRenderer(
            total_timesteps=total_timesteps,
            storage_dir=storage_dir,
            mode=mode
        )
        self.save_path = utils.build_rewards_path(storage_dir, mode)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        self.renderer.render()
        self.renderer.save(file_path=self.save_path)
