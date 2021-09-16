import os
import sys
from typing import List

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from yacht import utils, Mode
from yacht.data.renderers import RewardsRenderer
from yacht.environments import MetricsVecEnvWrapper
from yacht.logger import Logger
from yacht.utils import build_best_metric_checkpoint_file_name


class LoggerCallback(BaseCallback):
    def __init__(
            self,
            logger: Logger,
            log_frequency: int,
            total_timesteps: int,
            verbose: int = 0
    ):
        super().__init__(verbose)

        self.logger = logger
        self.total_timesteps = total_timesteps
        self.log_frequency = log_frequency

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_frequency == 0:
            self.logger.info(f'Timestep [{self.num_timesteps} / {self.total_timesteps}]')

            time_length_info = self.aggregate_time_lengths()
            self.logger.log(time_length_info, Logger.ONLY_CLOUD)

        return True

    def aggregate_time_lengths(self) -> dict:
        time_length_info = {
            'time/data_time_length': 0,
            'time/env_time_length': 0,
            'time/agent_time_length': 0
        }
        for info in self.training_env.unwrapped.buf_infos:
            time_length_info['time/data_time_length'] += info['data_time_length']
            time_length_info['time/env_time_length'] += info['env_time_length']
            time_length_info['time/agent_time_length'] += info['agent_time_length']

        num_envs = self.training_env.num_envs
        for k, v in time_length_info.items():
            time_length_info[k] = v / num_envs

        return time_length_info


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


class MetricsEvalCallback(EvalCallback):
    def __init__(
            self,
            eval_env: MetricsVecEnvWrapper,
            metrics_to_save_best_on: List[str],
            mode: Mode,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            best_model_save_path: str = None,
            deterministic: bool = True,
            render: bool = False,
            verbose: int = 1,
            warn: bool = True,
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=None,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn
        )

        self.mode = mode
        self.metrics_to_save_best_on = metrics_to_save_best_on
        self.best_metrics_results = {key: -sys.maxsize for key in self.metrics_to_save_best_on}

    def _on_step(self) -> bool:
        super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            step_mean_metrics = self.eval_env.mean_metrics
            for metric in self.metrics_to_save_best_on:
                metric_mean_value = step_mean_metrics[metric]
                if metric_mean_value > self.best_metrics_results[metric]:
                    if self.verbose > 0:
                        print(f'New best mean {metric}!')
                    if self.best_model_save_path is not None:
                        self.model.save(
                            os.path.join(self.best_model_save_path, build_best_metric_checkpoint_file_name(metric))
                        )
                    self.best_metrics_results[metric] = metric_mean_value

        return True
