import random
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class BaseClassicAgent(ABC):
    class DummyPolicy:
        # TODO: Move the predict logic to the policy.
        def train(self):
            pass

        def eval(self):
            pass

    def __init__(self, env: VecEnv):
        self.env = env
        self.policy = self.DummyPolicy()  # Quack like a real agent.

    def predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if state is not None and (state == 0).any():
            self.reset()

        actions = self._predict(observation, deterministic)

        return actions, np.ones_like(actions)

    @abstractmethod
    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class OnceAgent(BaseClassicAgent, ABC):
    def __init__(self, env: VecEnv, buy_tick: int):
        super().__init__(env=env)

        self.buy_tick = buy_tick
        self.current_tick = self.env.envs[0].unwrapped.start_tick

    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        if self.current_tick == self.buy_tick:
            total_close_prices = observation['1d'][:, -1, 0, :, 0]
            total_cash_positions = observation['env_features'][:, -1, 0]

            total_cash_per_asset = total_cash_positions / num_assets
            total_cash_per_asset = np.tile(total_cash_per_asset, reps=num_assets)
            total_cash_per_asset = total_cash_per_asset.reshape(num_envs, num_assets)
            actions = total_cash_per_asset / total_close_prices
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        self.current_tick += 1

        return actions

    def reset(self) -> None:
        self.current_tick = self.env.envs[0].unwrapped.start_tick


class OnceBeginningAgent(OnceAgent):
    def __init__(self, env: VecEnv):
        super().__init__(env=env, buy_tick=0)


class OnceRandomAgent(OnceAgent):
    def __init__(self, env: VecEnv):
        start_tick = env.envs[0].unwrapped.start_tick
        end_tick = env.envs[0].unwrapped.end_tick - 1
        buy_tick = random.randint(start_tick, end_tick)

        super().__init__(env=env, buy_tick=buy_tick)

    def reset(self) -> None:
        super().reset()

        start_tick = self.env.envs[0].unwrapped.start_tick
        end_tick = self.env.envs[0].unwrapped.end_tick - 1
        self.buy_tick = random.randint(start_tick, end_tick)


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

    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        if self.cash_distribution_per_ticker is None:
            total_cash_positions = observation['env_features'][:, -1, 0]
            self.cash_distribution_per_tick = total_cash_positions / self.num_buying_times

            self.cash_distribution_per_ticker = self.cash_distribution_per_tick / num_assets
            self.cash_distribution_per_ticker = self.cash_distribution_per_ticker.reshape(num_envs, 1)

        if self.bought_n_times < self.num_buying_times and self.current_tick >= self.buying_ticks[self.bought_n_times]:
            total_close_prices = observation['1d'][:, -1, 0, :, 0]
            actions = self.cash_distribution_per_ticker / total_close_prices

            self.bought_n_times += 1
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        self.current_tick += 1

        return actions

    def reset(self) -> None:
        assert self.bought_n_times == self.num_buying_times

        self.cash_distribution_per_ticker = None
        self.bought_n_times = 0
        self.current_tick = self.start_tick
