import os
import random
from abc import abstractmethod
from typing import Dict, List, Optional

import gym
from gym import spaces
import numpy as np

from yacht.data.datasets import TradingDataset
from yacht.environments.action_schemas import ActionSchema
from yacht.environments.enums import Position
from yacht.environments.reward_schemas import RewardSchema


class TradingEnv(gym.Env):
    def __init__(
            self,
            dataset: TradingDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            seed: int = 0
    ):
        from yacht.data.renderers import TradingRenderer

        self.seed(seed=seed)
        self.dataset = dataset
        self.window_size = dataset.window_size
        self.prices = dataset.get_prices()
        self.reward_schema = reward_schema
        self.action_schema = action_schema

        # spaces
        self.action_space = self.action_schema.get_action_space()
        self.observation_space = spaces.Dict(self.get_observation_space())
        assert self.observation_space['env_features'] is None or len(self.observation_space['env_features'].shape) == 1

        # Track
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 2  # Another -1 because the reward is calculated with one step further.
        self._done = None
        self._current_tick = None
        self._current_state = None
        self._current_reward = None
        self._last_position = None
        self._total_value = None
        self._total_profit = None
        self.history = None

        self.reset()

        # Rendering
        self.renderer = TradingRenderer(
            data=self.prices,
            start=dataset.start,
            end=dataset.end
        )

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    @property
    def current_profit(self):
        return self._total_value

    @property
    def intervals(self) -> List[str]:
        return self.dataset.intervals

    @property
    def observation_env_features_len(self) -> int:
        return self.observation_space['env_features'].shape[0] \
            if self.observation_space['env_features'] is not None \
            else 0

    def set_dataset(self, dataset: TradingDataset):
        from yacht.data.renderers import TradingRenderer

        self.dataset = dataset
        self.prices = self.dataset.get_prices()

        # spaces
        self.observation_space = spaces.Dict(self.get_observation_space())
        assert self.observation_space['env_features'] is None or len(self.observation_space['env_features'].shape) == 1

        # episode
        self._end_tick = len(self.prices) - 1

        # Rendering
        self.renderer = TradingRenderer(
            data=self.prices,
            start=dataset.start,
            end=dataset.end
        )

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._current_state = None
        self._current_reward = None
        self._last_position = None
        self._total_value = 0.
        self.history = {}

        self.reward_schema.reset()

        return self.get_next_observation()

    def step(self, action: np.array):
        # TODO: Is this ok ?
        if self._done is True:
            return self._current_state, self._current_reward, self._done, {}
        else:
            self._current_tick += 1

            if self._current_tick == self._end_tick:
                self._done = True

            action = self.action_schema.get_value(action)

            self._current_reward = self.reward_schema.calculate_reward(
                action=action,
                dataset=self.dataset,
                current_index=self._current_tick - 1
            )
            self.update_total_value(action)

            self._current_state = self.get_next_observation()

            info = self.create_info(action)
            self.update_history(info)

            return self._current_state, self._current_reward, self._done, info

    def create_info(self, action: np.array) -> dict:
        action = action.item()
        position = Position.build(position=action)

        info = dict(
            step=self._current_tick,
            action=action,
            position=position,
            reward=self._current_reward,
            total_value=self._total_value,
            max_possible_value=(self._current_tick - self._start_tick) * self.action_schema.max_units_per_asset,
            total_value_completeness=round(self._total_value / self.max_possible_profit(stateless=True), 2)
        )

        return info

    def get_observation_space(self) -> Dict[str, Optional[spaces.Space]]:
        observation_space = self.dataset.get_external_observation_space()
        observation_space['env_features'] = None

        return observation_space

    def get_next_observation(self) -> Dict[str, np.array]:
        observation = self.dataset[self._current_tick]

        return observation

    def update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
            # There are no positions & actions before the starting point.
            self.history['position'] = (self.window_size - 1) * [np.nan]
            self.history['action'] = (self.window_size - 1) * [0]

        # For easier processing add a position only when it is changing.
        position = info.pop('position')
        if self._last_position != position:
            self.history['position'].append(position)
            self._last_position = position
        else:
            self.history['position'].append(np.nan)

        for key, value in info.items():
            self.history[key].append(value)

    @abstractmethod
    def render(self, mode='human', name='trades.png'):
        pass

    def render_all(self, title, name='trades.png'):
        self.renderer.render(
            title=title,
            save_file_path=os.path.join(self.dataset.storage_dir, name),
            positions=self.history['position'],
            actions=self.history['action']
        )

    def close(self):
        pass

    def calculate_reward(self, action):
        raise NotImplementedError

    def update_total_value(self, action):
        raise NotImplementedError

    def max_possible_profit(self, stateless=True):
        raise NotImplementedError()
