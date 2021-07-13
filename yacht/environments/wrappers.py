from typing import Dict, List

import gym
import numpy as np
import torch
import wandb
from gym import spaces

from yacht.agents.misc import unflatten_observations
from yacht.environments import TradingEnv, Mode


class MultipleTimeFrameDictToBoxWrapper(gym.Wrapper):
    def __init__(self, env: TradingEnv):
        super().__init__(env)

        self.observation_space = self._compute_flattened_observation_space()

    def _compute_flattened_observation_space(self) -> spaces.Box:
        current_observation_space = self.env.observation_space
        window_size = current_observation_space['1d'].shape[0]
        feature_size = current_observation_space['1d'].shape[2]
        bars_size = sum([v.shape[1] for k, v in current_observation_space.spaces.items() if k != 'env_features'])

        env_features_space = current_observation_space['env_features']
        env_features_size = env_features_space.shape[0] if env_features_space is not None else 0

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, bars_size + env_features_size, feature_size),
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

        env_features = np.array(observation['env_features'], dtype=np.float32)
        env_features = env_features.reshape((1, -1, 1))
        env_features = np.tile(
            env_features,
            (flattened_observation.shape[0], env_features.shape[1], flattened_observation.shape[2])
        )
        flattened_observation = np.concatenate([
            flattened_observation,
            env_features
        ], axis=1)

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

        info_to_log = dict()
        if is_done and episode_metrics:
            info_to_log['total_value'] = info['total_value']
            info_to_log['num_longs'] = info['num_longs']
            info_to_log['num_shorts'] = info['num_shorts']
            info_to_log['num_holds'] = info['num_holds']
            info_to_log['profit_hits'] = info['profit_hits']
            info_to_log['loss_misses'] = info['loss_misses']
            info_to_log['hit_ratio'] = info['hit_ratio']

            # TODO: Log more metrics after we understand them.
            info_to_log['episode_metrics'] = {
                'Annual return': episode_metrics['Annual return'],
                'Cumulative returns': episode_metrics['Cumulative returns'],
                'Annual volatility': episode_metrics['Annual volatility'],
                'Sharpe ratio': episode_metrics['Sharpe ratio']
            }

            # Translate the keys for easier understanding
            info_to_log['episode'] = {
                'reward': episode_data['r'],
                'length': episode_data['l'],
                'seconds': episode_data['t']
            }

        wandb.log({
            self.mode.value: info_to_log
        })

        return obs, reward, terminal, info
