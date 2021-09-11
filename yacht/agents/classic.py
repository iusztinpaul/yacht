from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


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
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        if state is not None and (state == 0).any():
            self.bought = False

        if self.bought is False:
            total_close_prices = observation['1d'][:, -1, 0, :, 0]
            total_cash_positions = observation['env_features'][:, -1, 0]

            # TODO: Adapt cash distribution by market cap.
            total_cash_per_asset = total_cash_positions / num_assets
            total_cash_per_asset = np.tile(total_cash_per_asset, reps=num_assets)
            total_cash_per_asset = total_cash_per_asset.reshape(num_envs, num_assets)
            actions = total_cash_per_asset / total_close_prices

            self.bought = True
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        return actions, np.ones(shape=(num_envs, num_assets))


class DCFAgent(BaseClassicAgent):
    def __init__(self, env: VecEnv):
        super().__init__(env=env)

        self.start_tick = self.env.envs[0].unwrapped.start_tick
        self.end_tick = self.env.envs[0].unwrapped.end_tick
        self.current_tick = self.start_tick
        self.num_buying_times = 4
        self.buy_period = (self.end_tick - self.start_tick) // self.num_buying_times
        self.buying_ticks = [v for v in range(self.start_tick, self.end_tick, self.buy_period)]
        self.bought_n_times = 0
        self.cash_distribution_per_tick = None
        self.cash_distribution_per_ticker = None

    def predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        if state is not None and (state == 0).any():
            assert self.bought_n_times == self.num_buying_times

            self.cash_distribution_per_ticker = None
            self.bought_n_times = 0
            self.current_tick = self.start_tick

        if self.cash_distribution_per_ticker is None:
            total_cash_positions = observation['env_features'][:, -1, 0]
            self.cash_distribution_per_tick = total_cash_positions / self.num_buying_times

            # TODO: Adapt cash distribution by market cap.
            num_assets = observation['1d'].shape[3]
            self.cash_distribution_per_ticker = self.cash_distribution_per_tick / num_assets
            self.cash_distribution_per_ticker = np.tile(self.cash_distribution_per_ticker, num_assets)

        if self.bought_n_times < self.num_buying_times and self.current_tick >= self.buying_ticks[self.bought_n_times]:
            total_close_prices = observation['1d'][:, -1, 0, :, 0]
            actions = self.cash_distribution_per_ticker / total_close_prices

            self.bought_n_times += 1
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        self.current_tick += 1

        return actions, np.ones(shape=(num_envs, num_assets))
