import os
from typing import Union

import gym
from gym import spaces
import numpy as np

from yacht.data.datasets import TradingDataset
from yacht.environments.action_schemas import ActionSchema
from yacht.environments.enums import Position
from yacht.environments.reward_schemas import RewardSchema


class TradingEnv(gym.Env):
    def __init__(self, dataset: TradingDataset, reward_schema: RewardSchema, action_schema: ActionSchema):
        from yacht.data.renderers import TradingRenderer

        self.seed()
        self.dataset = dataset
        self.window_size = dataset.window_size
        self.prices = dataset.get_prices()
        self.reward_schema = reward_schema
        self.action_schema = action_schema

        # spaces
        self.action_space = self.action_schema.get_action_space()
        self.observation_space_shape = (self.window_size, *self.dataset.get_item_shape())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.observation_space_shape,
            dtype=np.float32
        )

        # Track
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 2  # Another -1 because the reward is calculated with one step further.
        self._done = None
        self._current_tick = None
        self._current_state = None
        self._current_reward = None
        self._total_reward = None
        self._total_profit = None
        self.history = None

        self.reset()

        # Rendering
        self.renderer = TradingRenderer(
            prices=self.prices.loc[:, 'Close'],
            start=dataset.start,
            end=dataset.end
        )

    def set_dataset(self, dataset: TradingDataset):
        from yacht.data.renderers import TradingRenderer

        self.dataset = dataset
        self.prices = self.dataset.get_prices()

        # spaces
        self.observation_space_shape = (self.window_size, *self.dataset.get_item_shape())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.observation_space_shape,
            dtype=np.float32
        )

        # episode
        self._end_tick = len(self.prices) - 1

        # Rendering
        self.renderer = TradingRenderer(
            prices=self.prices.loc[:, 'Close'],
            start=dataset.start,
            end=dataset.end
        )

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._current_state = None
        self._current_reward = None
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self.history = {}

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
            position = Position.build(position=action)

            self._current_reward = self.reward_schema.calculate_reward(
                action=action,
                dataset=self.dataset,
                current_index=self._current_tick - 1
            )
            self._total_reward += self._current_reward

            self.update_profit(action)

            self._current_state = self.get_next_observation()

            info = dict(
                total_reward=self._total_reward,
                total_profit=self._total_profit,
                action=action,
                position=position,
                reward=self._current_reward,
            )
            self._update_history(info)

            return self._current_state, self._current_reward, self._done, info

    def get_next_observation(self) -> np.array:
        observation, _ = self.dataset[self._current_tick]

        return observation

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}
            self.history['render_position'] = (self.window_size - 1) * [None]

        if len(self.history['position']) == 0 or self.history['position'][-1] != info['position']:
            self.history['render_position'].append(info['position'])
        else:
            self.history['render_position'].append(None)

        for key, value in info.items():
            self.history[key].append(value)

    def render(self, mode='live'):
        self.renderer.render(self.history['render_position'])
        self.renderer.pause()

    def render_all(self, show=True):
        self.renderer.render(self.history['render_position'])
        if show:
            self.renderer.show()

    def close(self):
        self.renderer.close()

    def save_rendering(self, name='trades.png'):
        self.renderer.save(
            os.path.join(self.dataset.storage_dir, name)
        )

    def calculate_reward(self, action):
        raise NotImplementedError

    def update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):
        raise NotImplementedError()
