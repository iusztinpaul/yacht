import os
import sys
from collections import Counter
from typing import List, Optional

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
            time_length_info['timings_step'] = self.num_timesteps
            self.logger.log(time_length_info, Logger.SKIP_COUT)

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
    class DummyCallback(BaseCallback):
        def _on_step(self) -> bool:
            return True

    def __init__(
            self,
            eval_env: MetricsVecEnvWrapper,
            metrics_to_save_best_on: List[str],
            plateau_max_n_steps: int,
            mode: Mode,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            log_path: str = None,
            best_model_save_path: str = None,
            deterministic: bool = True,
            render: bool = False,
            logger: Optional[Logger] = None,
            verbose: int = 1,
            warn: bool = True,
    ):
        if verbose:
            assert logger is not None, 'If "verbose=1" you should pass the "logger" argument.'

        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=self.DummyCallback(),
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
        self.apply_plateau = plateau_max_n_steps > 0
        self.plateau_max_n_steps = plateau_max_n_steps
        self.best_metrics_results = {key: -sys.maxsize for key in self.metrics_to_save_best_on}
        self.plateau_metrics_counter = Counter()
        self.found_any_new = False

    def _on_step(self) -> bool:
        super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            step_mean_metrics = self.eval_env.mean_metrics
            for metric in self.metrics_to_save_best_on:
                metric_mean_value = step_mean_metrics[metric]
                if metric_mean_value > self.best_metrics_results[metric]:
                    self.found_any_new = True
                    self.plateau_metrics_counter[metric] = 0

                    if self.verbose > 0:
                        self.logger.log(f'New best mean for: {metric}!')
                        self.logger.record(f'{self.mode.value}-max/{metric}', metric_mean_value)
                        self.logger.record(f'{self.mode.value}-max/{metric}-step', self.n_calls)
                    if self.best_model_save_path is not None:
                        self.model.save(
                            os.path.join(self.best_model_save_path, build_best_metric_checkpoint_file_name(metric))
                        )
                    self.best_metrics_results[metric] = metric_mean_value
                else:
                    self.plateau_metrics_counter[metric] += 1

            if self.verbose > 0 and self.found_any_new is True:
                self.logger.dump()
                self.found_any_new = False

            if self.apply_plateau:
                # If any metric we are listing to did not had a new best value for 'max_n_steps' end training.
                metrics_plateau = [
                    plateau_steps <= self.plateau_max_n_steps for plateau_steps in self.plateau_metrics_counter.values()
                ]
                continue_training = any(metrics_plateau)
                if continue_training is False:
                    self.logger.info('Stopped training because of plateauing metrics...')

                return continue_training

        return True

    def _on_event(self) -> bool:
        state = super()._on_event()

        self.found_any_new = True

        self.logger.log(f'New best mean for: reward!')
        self.logger.record(f'{self.mode.value}-max/reward', self.best_mean_reward)
        self.logger.record(f'{self.mode.value}-max/reward-step', self.n_calls)

        return state
