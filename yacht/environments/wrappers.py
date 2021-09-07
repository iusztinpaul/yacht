from abc import ABC
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
            log_std: bool = False,
    ):
        super().__init__(env)

        self.n_metrics_episodes = n_metrics_episodes
        self.logger = logger
        self.mode = mode
        self.log_std = log_std

        self.metrics: List[dict] = []

        self._mean_metrics = dict()
        self._std_metrics = dict()

    @property
    def mean_metrics(self) -> dict:
        return self._mean_metrics

    @property
    def std_metrics(self) -> dict:
        return self._std_metrics

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
                    self.metrics.append(self._extract_metrics(info=info[idx]))

            if len(self.metrics) >= self.n_metrics_episodes:
                mean_metrics_over_envs, std_metrics_over_envs = self._compute_mean_std(
                    infos=self.metrics[-self.n_metrics_episodes:]
                )
                self._mean_metrics = self._flatten_keys(mean_metrics_over_envs)
                self._std_metrics = self._flatten_keys(std_metrics_over_envs)

                self.logger.log(self._mean_metrics)
                if self.log_std:
                    self.logger.log(self._std_metrics)

        return obs, reward, done, info

    def _flatten_keys(self, metrics_to_log: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        flattened_dict = dict()
        for metric_name, metric_value in metrics_to_log.items():
            flattened_dict[f'{self.mode.value}/{metric_name}'] = metric_value

        return flattened_dict

    @classmethod
    def _extract_metrics(cls, info: dict) -> dict:
        episode_metrics = info['episode_metrics']
        episode_data = info['episode']

        metrics_to_log = {
            'reward': episode_data['r'],
            'annual_return': episode_metrics['annual_return'],
            'cumulative_returns': episode_metrics['cumulative_returns'],
            'sharpe_ratio': episode_metrics['sharpe_ratio'],
            'max_drawdown': episode_metrics['max_drawdown'],
        }
        if episode_metrics.get('buy_pa'):
            metrics_to_log['buy_pa'] = episode_metrics['buy_pa']
        if episode_metrics.get('sell_pa'):
            metrics_to_log['sell_pa'] = episode_metrics['sell_pa']

        return metrics_to_log

    @classmethod
    def _compute_mean_std(cls, infos: List[dict]) -> Tuple[dict, dict]:
        aggregated_metrics: Dict[str, list] = defaultdict(list)
        for env_metrics in infos:
            for metric_name, metric_value in env_metrics.items():
                aggregated_metrics[metric_name].append(metric_value)

        mean_metrics: Dict[str, np.ndarray] = dict()
        std_metrics: Dict[str, np.ndarray] = dict()
        for metric_name, metric_values in aggregated_metrics.items():
            metric_values = np.array(metric_values, dtype=np.float32)

            mean_metrics[metric_name] = np.mean(metric_values)
            std_metrics[metric_name] = np.std(metric_values)

        return mean_metrics, std_metrics
