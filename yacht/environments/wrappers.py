from typing import Dict, List

import gym
import numpy as np
import torch
import wandb
from gym import spaces

from yacht.agents.misc import unflatten_observations
from yacht.environments import BaseAssetEnvironment
from yacht import Mode


class MultiFrequencyDictToBoxWrapper(gym.Wrapper):
    def __init__(self, env: BaseAssetEnvironment):
        super().__init__(env)

        self.observation_space = self._compute_flattened_observation_space()

    def _compute_flattened_observation_space(self) -> spaces.Box:
        current_observation_space = self.env.observation_space
        window_size = current_observation_space['1d'].shape[0]
        feature_size = current_observation_space['1d'].shape[2]
        bars_size = sum([v.shape[1] for k, v in current_observation_space.spaces.items() if k != 'env_features'])

        env_features_space = current_observation_space['env_features']
        env_features_size = env_features_space.shape[1] if env_features_space is not None else 0

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, bars_size, feature_size + env_features_size),
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

        # Concatenate env_features which are features at the window level.
        env_features = observation['env_features']
        window_size, feature_size = env_features.shape
        env_features = env_features.reshape((window_size, 1, feature_size))
        env_features = np.tile(
            env_features,
            (1, flattened_observation.shape[1], 1)
        )
        flattened_observation = np.concatenate([
            flattened_observation,
            env_features
        ], axis=-1)

        return flattened_observation

    @classmethod
    def unflatten_observation(cls, intervals: List[str], observations: np.array) -> np.array:
        observations = torch.from_numpy(observations)
        observations = unflatten_observations(observations, intervals)
        observations = observations.numpy()

        return observations


class WandBWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, mode: Mode):
        super().__init__(env)

        self.mode = mode

    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)

        is_done = info['done']
        episode_metrics = info.get('episode_metrics', False)
        episode_data = info.get('episode', False)

        if not self.mode.is_trainable():
            if is_done and episode_metrics:
                long_short_ratio = episode_metrics['num_longs'] / episode_metrics['num_shorts'] \
                    if episode_metrics['num_shorts'] != 0 \
                    else 1.

                info_to_log = {
                        'reward': episode_data['r'],
                        'annual_return': episode_metrics['annual_return'],
                        'cumulative_returns': episode_metrics['cumulative_returns'],
                        'sharpe_ratio': episode_metrics['sharpe_ratio'],
                        'max_drawdown': episode_metrics['max_drawdown'],
                        'LSR': long_short_ratio
                    }
                if episode_metrics.get('buy_pa'):
                    info_to_log['buy_pa'] = episode_metrics['buy_pa']
                if episode_metrics.get('sell_pa'):
                    info_to_log['sell_pa'] = episode_metrics['sell_pa']

                wandb.log({
                    self.mode.value: info_to_log
                })

        return obs, reward, terminal, info
