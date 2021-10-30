from collections import defaultdict
from typing import Dict, List, Any, Optional

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
        env_features_size += 1  # Add 1 because of the padding meta information.

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

        # Pad env_features in the case that window_size_data != window_size_env e.g.
        # Add that metadata to the env_features.
        padding_size = flattened_observation.shape[0] - env_features.shape[0]
        env_features = np.pad(env_features, ((0, padding_size), (0, 0), (0, 0)))
        padding_size = np.full(shape=(flattened_observation.shape[0], 1, 1), fill_value=padding_size, dtype=np.int32)
        env_features = np.concatenate([
            env_features,
            padding_size
        ], axis=-1)

        flattened_observation = np.concatenate([
            flattened_observation,
            env_features
        ], axis=-1)

        return flattened_observation


class MetricsVecEnvWrapper(VecEnvWrapper):
    def __init__(
            self,
            venv: VecEnv,
            n_metrics_episodes: int,
            logger: Logger,
            mode: Mode,
            metrics_to_log: List[str],
            extra_stats_metrics: Optional[List[str]] = None,
            load_best_metric: Optional[str] = None
    ):
        if mode.is_best_metric():
            assert load_best_metric is not None

        super().__init__(venv)

        self.n_metrics_episodes = n_metrics_episodes
        self.logger = logger
        self.mode = mode
        self.metrics_to_log = metrics_to_log
        self.extra_metrics_to_log = extra_stats_metrics if extra_stats_metrics is not None else []
        self.load_best_metric = load_best_metric

        self.metrics: List[dict] = []
        self.metric_statistics = {
            'mean': dict(),
            'median': dict(),
            'std': dict()
        }
        # Define an internal step to be able to compare metrics easier.
        self.num_step = 0

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
                metric_statistics[self.mode.to_step_key()] = self.num_step
                self.logger.log(metric_statistics)

                self.metrics = []

        self.num_step += self.venv.num_envs

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

    def flatten_dict(self, metrics_to_log: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        flattened_dict = dict()
        for per_statistic_values in metrics_to_log.values():
            for metric_name, metric_value in per_statistic_values.items():
                flattened_dict[self._prefix_key(metric_name)] = metric_value

        return flattened_dict

    def _prefix_key(self, key: str) -> str:
        if self.mode.is_best_metric():
            return f'{self.mode.value}-{self.load_best_metric}/{key}'

        return f'{self.mode.value}/{key}'

    def extract_metrics(self, info: dict) -> dict:
        episode_metrics = info['episode_metrics']
        episode_data = info['episode']

        metrics_to_log = {
            'reward': episode_data['r'],
        }
        for metric in self.metrics_to_log:
            if episode_metrics.get(metric) is not None:
                metrics_to_log[metric] = episode_metrics[metric]

        return metrics_to_log

    def compute_metrics_statistics(self, metrics: List[dict]) -> Dict[str, dict]:
        aggregated_metrics: Dict[str, list] = defaultdict(list)
        for env_metrics in metrics:
            for metric_name, metric_value in env_metrics.items():
                aggregated_metrics[metric_name].append(metric_value)

        metric_statistics = {
            'mean': dict(),
            'median': dict(),
            'std': dict()
        }
        for metric_name, metric_values in aggregated_metrics.items():
            metric_values = np.array(metric_values, dtype=np.float32)

            metric_statistics['mean'][metric_name] = np.mean(metric_values)
            # We don't want to clutter the board with redundant information. Log only essential metrics for non-mean.
            if metric_name in self.extra_metrics_to_log:
                metric_statistics['median'][f'median-{metric_name}'] = np.median(metric_values)
                metric_statistics['std'][f'std-{metric_name}'] = np.std(metric_values)

        return metric_statistics
