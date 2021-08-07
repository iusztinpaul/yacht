from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from yacht.environments import BaseAssetEnvironment


class BaseClassicAgent(ABC):
    def __init__(self, env: VecEnv):
        self.env = env

    @abstractmethod
    def predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pass


class BuyAndHoldAgent(BaseClassicAgent):
    def __init__(self, env: VecEnv):
        super().__init__(env=env)

        self.bought = False

    def predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.bought is False:
            total_close_prices = observation['1d'][:, -1, 0, 0]
            total_cash_positions = observation['env_features'][:, -1, 0]
            actions = total_cash_positions / total_close_prices

            self.bought = True
        else:
            batch_size = observation['1d'].shape[0]

            actions = np.zeros(shape=(batch_size, ))

        actions = np.expand_dims(actions, axis=1)

        return actions, None


class DCFAgent(BaseClassicAgent):
    def __init__(self, env: VecEnv):
        super().__init__(env=env)

        start_tick = self.env.envs[0].unwrapped.start_tick
        end_tick = self.env.envs[0].unwrapped.end_tick
        self.time_horizon = end_tick - start_tick
        self.cash_distribution_per_tick = None

    def predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.cash_distribution_per_tick is None:
            total_cash_positions = observation['env_features'][:, -1, 0]
            self.cash_distribution_per_tick = total_cash_positions / self.time_horizon

        total_close_prices = observation['1d'][:, -1, 0, 0]
        actions = self.cash_distribution_per_tick / total_close_prices
        actions = np.expand_dims(actions, axis=1)

        return actions, None
