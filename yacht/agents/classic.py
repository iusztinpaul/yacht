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
        self.total_units = -1

    def predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.bought is False:
            total_close_prices = observation['1d'][:, -1, 0, 0]
            total_cash_positions = observation['env_features'][:, -1, 0]

            action_schema = self.env.envs[0].unwrapped.action_schema

            self.total_units = total_cash_positions / total_close_prices
            actions = self.total_units / action_schema.max_units_per_asset

            self.bought = True
        else:
            batch_size = observation['1d'].shape[0]

            actions = np.zeros(shape=(batch_size, ))

        actions = np.expand_dims(actions, axis=1)

        return actions, None
