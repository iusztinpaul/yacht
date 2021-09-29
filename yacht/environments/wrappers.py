from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional, Any

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import VecEnvWrapper, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from yacht.environments import BaseAssetEnvironment
from yacht import Mode


class MultiFrequencyDictToBoxWrapper(gym.Wrapper):
    def __init__(self, env: BaseAssetEnvironment):
        super().__init__(env)

        self.observation_space = self._compute_flattened_observation_space()

    def _compute_flattened_observation_space(self) -> spaces.Box:
        current_observation_space = self.env.observation_space
        num_assets = current_observation_space['1d'].shape[2]
        window_size = current_observation_space['1d'].shape[0]
        feature_size = current_observation_space['1d'].shape[3]
        bars_size = sum([v.shape[1] for k, v in current_observation_space.spaces.items() if k != 'env_features'])

        env_features_space = current_observation_space['env_features']
        env_features_size = env_features_space.shape[1] if env_features_space is not None else 0

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, bars_size, feature_size * num_assets + env_features_size),
            dtype=np.float32
        )

    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)

        return self.flatten_observation(obs), reward, terminal, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        return self.flatten_observation(obs)

    def flatten_observation(self, observation: Dict[str, np.array]) -> np.array:
        intervals = self.env.intervals
        flattened_observation = [observation[interval] for interval in intervals]
        flattened_observation = np.concatenate(flattened_observation, axis=1)
        window_size, bars_size, _, _ = flattened_observation.shape
        flattened_observation = flattened_observation.reshape((window_size, bars_size, -1))

        # Concatenate env_features, which are features at the window level.
        env_features = observation['env_features']
        window_size, feature_size = env_features.shape
        env_features = env_features.reshape((window_size, 1, feature_size))
        # Env features are the same at the bar level.
        env_features = np.tile(
            env_features,
            (1, flattened_observation.shape[1], 1)
        )
        flattened_observation = np.concatenate([
            flattened_observation,
            env_features
        ], axis=-1)

        return flattened_observation


class MetricsVecEnvWrapper(VecEnvWrapper, ABC):
    def __init__(
            self,
            env: VecEnv,
            n_metrics_episodes: int,
            logger: Logger,
            mode: Mode,
            extra_metrics_to_log: List[str]
    ):
        super().__init__(env)

        self.n_metrics_episodes = n_metrics_episodes
        self.logger = logger
        self.mode = mode
        self.extra_metrics_to_log = extra_metrics_to_log

        self.metrics: List[dict] = []
        self.metric_statistics = {
            'mean': dict(),
            'median': dict(),
            'std': dict(),
            'third_quartile': dict()
        }

    @property
    def mean_metrics(self) -> dict:
        return self.metric_statistics['mean']

    @property
    def median_metrics(self) -> dict:
        return self.metric_statistics['median']

    @property
    def std_metrics(self) -> dict:
        return self.metric_statistics['std']

    @property
    def third_quartile_metrics(self) -> dict:
        return self.metric_statistics['third_quartile']

    def reset(self) -> np.ndarray:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, info = self.venv.step_wait()

        if not self.mode.is_trainable():
            # Persist the metrics from the finished environments.
            if done.any():
                done_indices = np.where(done)[0]
                for idx in done_indices:
                    self.metrics.append(self.extract_metrics(info=info[idx]))

            if len(self.metrics) >= self.n_metrics_episodes:
                self.metric_statistics = self.compute_metrics_statistics(metrics=self.metrics)
                self.metric_statistics['mean'].update(self.computed_aggregated_metrics())

                metric_statistics = self.flatten_dict(self.metric_statistics)
                self.logger.log(metric_statistics)

                self.metrics = []

        return obs, reward, done, info

    def computed_aggregated_metrics(self) -> dict:
        """
            Compute metrics were we need results from multiple environment runs.
        Returns:
            Metrics in form of a dictionary.
        """
        from yacht import evaluation

        glr_ratio = evaluation.compute_glr_ratio(pa_values=[env_metric['PA'] for env_metric in self.metrics])

        return {
            'GLR': glr_ratio
        }

    def flatten_dict(self, metrics_to_log: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        flattened_dict = dict()
        for per_statistic_values in metrics_to_log.values():
            for metric_name, metric_value in per_statistic_values.items():
                flattened_dict[self._prefix_key(metric_name)] = metric_value

        return flattened_dict

    def _prefix_key(self, key: str) -> str:
        return f'{self.mode.value}/{key}'

    @classmethod
    def extract_metrics(cls, info: dict) -> dict:
        episode_metrics = info['episode_metrics']
        episode_data = info['episode']

        metrics_to_log = {
            'reward': episode_data['r'],
            'annual_return': episode_metrics['annual_return'],
            'cumulative_returns': episode_metrics['cumulative_returns'],
            'sharpe_ratio': episode_metrics['sharpe_ratio'],
            'max_drawdown': episode_metrics['max_drawdown'],
        }
        if episode_metrics.get('PA') is not None:
            metrics_to_log['PA'] = episode_metrics['PA']
        if episode_metrics.get('cash_used_on_last_tick') is not None:
            metrics_to_log['cash_used_on_last_tick'] = episode_metrics['cash_used_on_last_tick']

        return metrics_to_log

    def compute_metrics_statistics(self, metrics: List[dict]) -> Dict[str, dict]:
        aggregated_metrics: Dict[str, list] = defaultdict(list)
        for env_metrics in metrics:
            for metric_name, metric_value in env_metrics.items():
                aggregated_metrics[metric_name].append(metric_value)

        metric_statistics = {
            'mean': dict(),
            'median': dict(),
            'std': dict(),
            'third_quartile': dict()
        }
        for metric_name, metric_values in aggregated_metrics.items():
            metric_values = np.array(metric_values, dtype=np.float32)

            metric_statistics['mean'][metric_name] = np.mean(metric_values)
            # We don't want to clutter the board with redundant information. Log only essential metrics for non-mean.
            if metric_name in self.extra_metrics_to_log:
                metric_statistics['median'][f'median-{metric_name}'] = np.median(metric_values)
                metric_statistics['std'][f'std-{metric_name}'] = np.std(metric_values)
                metric_statistics['third_quartile'][f'quantile-75-{metric_name}'] = np.quantile(metric_values, 0.75)

        return metric_statistics
