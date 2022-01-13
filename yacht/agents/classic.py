import random
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class BaseClassicAgent(ABC):
    class DummyPolicy:
        # TODO: Adapt the agents for sell logic.
        # TODO: Move the predict logic to the policy.
        def train(self):
            pass

        def eval(self):
            pass

    def __init__(self, env: VecEnv, window_size: int):
        self.env = env
        self.window_size = window_size
        self.policy = self.DummyPolicy()  # Quack like a real agent.

        self.start_tick = self.env.envs[0].unwrapped.start_tick
        self.end_tick = self.env.envs[0].unwrapped.end_tick
        self.current_tick = self.start_tick

    def predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            state: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if state is not None and (state == 0).any():
            self.reset()

        actions = self._predict(observation, deterministic)

        self.current_tick += 1

        return actions, np.ones_like(actions)

    @abstractmethod
    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        pass

    def reset(self) -> None:
        self.start_tick = self.env.envs[0].unwrapped.start_tick
        self.end_tick = self.env.envs[0].unwrapped.end_tick
        self.current_tick = self.start_tick


class OnceKnownTickerAgent(BaseClassicAgent, ABC):
    def __init__(self, env: VecEnv, window_size: int, buy_tick: int):
        super().__init__(env=env, window_size=window_size)

        self.buy_tick = buy_tick

    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        if self.current_tick == self.buy_tick:
            action_per_asset = 1 / num_assets
            actions = np.tile(action_per_asset, (num_envs, num_assets))
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        return actions


class OnceBeginningAgent(OnceKnownTickerAgent):
    def __init__(self, env: VecEnv, window_size: int):
        super().__init__(env=env, window_size=window_size, buy_tick=0)


class OnceRandomAgent(OnceKnownTickerAgent):
    def __init__(self, env: VecEnv, window_size: int):
        start_tick = env.envs[0].unwrapped.start_tick
        end_tick = env.envs[0].unwrapped.end_tick
        buy_tick = random.randint(start_tick, end_tick - 1)

        super().__init__(env=env, window_size=window_size, buy_tick=buy_tick)

    def reset(self) -> None:
        super().reset()

        self.buy_tick = random.randint(self.start_tick, self.end_tick - 1)


class EquallyDistributedInTimeAgent(BaseClassicAgent):
    def __init__(self, env: VecEnv, window_size: int):
        super().__init__(env=env, window_size=window_size)

        self.num_buying_times = 4
        self.action_per_tick = 1 / self.num_buying_times
        self.buy_period = (self.end_tick - self.start_tick) // self.num_buying_times
        self.buying_ticks = [v for v in range(self.start_tick, self.end_tick, self.buy_period)]
        self.bought_n_times = 0

        self.action_per_asset = None

    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        if self.action_per_asset is None:
            self.action_per_asset = self.action_per_tick / num_assets
            self.action_per_asset = np.tile(self.action_per_asset, (num_envs, num_assets))

        if self.bought_n_times < self.num_buying_times and self.current_tick >= self.buying_ticks[self.bought_n_times]:
            actions = self.action_per_asset

            self.bought_n_times += 1
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        return actions

    def reset(self) -> None:
        super().reset()

        assert self.bought_n_times == self.num_buying_times

        self.action_per_asset = None
        self.bought_n_times = 0


class TWAPAgent(BaseClassicAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tick_quantity = 1 / (self.end_tick - self.start_tick)

    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        tick_quantity_per_asset = self.tick_quantity / num_assets
        actions = np.tile(tick_quantity_per_asset, (num_envs, num_assets))

        return actions

    def reset(self):
        super().reset()

        self.tick_quantity = 1 / (self.end_tick - self.start_tick)


class VWAPAgent(BaseClassicAgent):
    # TODO: See why in same cases the agent stops at taking 5 actions.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tick_quantity = 0.25

    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]
        data = observation['1d']

        # typical_price = (High + Low + Close) / 3
        typical_price = (data[..., 0] + data[..., 2] + data[..., 3]) / 3
        volume = data[..., 4]
        typical_price = np.squeeze(typical_price, axis=2)
        volume = np.squeeze(volume, axis=2)
        vwap = ((typical_price * volume).sum(axis=1) + 1e-8) / (volume.sum(axis=1) + 1e-8)
        # TODO: Adapt buy logic for multiple assets.
        if (typical_price[:, -1, :] <= vwap).any():
            tick_quantity_per_asset = self.tick_quantity / num_assets
            actions = np.tile(tick_quantity_per_asset, (num_envs, num_assets))
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        return actions


class OnceInferredTickerAgent(BaseClassicAgent, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_tick = None

    def _predict(
            self,
            observation: Union[Dict[str, np.ndarray], np.ndarray],
            deterministic: bool = False,
    ) -> np.ndarray:
        num_envs = observation['1d'].shape[0]
        num_assets = observation['1d'].shape[3]

        if self.action_tick is None:
            # Teacher data is padded for ENV compatibility.
            # For classic methods the data is not flattened anymore, but the padding logic is still applied.
            data = observation['1d'][:, 1:self.end_tick]
            typical_price = (data[..., 0] + data[..., 1] + 2 * data[..., 2] + 2 * data[..., 3]) / 6
            typical_price = np.squeeze(typical_price, axis=2)
            typical_price = np.squeeze(typical_price, axis=2)
            self.action_tick = self.get_action_tick(typical_price)
        # TODO: Adapt buy logic for multiple assets.
        if (self.current_tick == self.action_tick).any():
            actions = np.tile(1 / num_assets, (num_envs, num_assets))
        else:
            actions = np.zeros(shape=(num_envs, num_assets))

        return actions

    @abstractmethod
    def get_action_tick(self, typical_price: np.ndarray) -> np.ndarray:
        pass

    def reset(self):
        super().reset()

        self.action_tick = None


class BestActionAgent(OnceInferredTickerAgent):
    def get_action_tick(self, typical_price: np.ndarray) -> np.ndarray:
        return typical_price.argmin(axis=1)


class WorstActionAgent(OnceInferredTickerAgent):
    def get_action_tick(self, typical_price: np.ndarray) -> np.ndarray:
        return typical_price.argmax(axis=1)
