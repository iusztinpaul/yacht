import datetime
import logging
import random
from abc import abstractmethod
from typing import Dict, List, Optional

import gym
import pandas as pd
from gym import spaces
import numpy as np

from yacht import utils
from yacht.data.datasets import TradingDataset
from yacht.environments.action_schemas import ActionSchema
from yacht.environments.enums import Position
from yacht.environments.reward_schemas import RewardSchema


logger = logging.getLogger(__file__)


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
        self._tick_t = None
        self._s_t = None
        self._a_t = None
        self._position_previous_t = None
        self._r_t = None
        self._total_value = None
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

    def get_observation_space(self) -> Dict[str, Optional[spaces.Space]]:
        observation_space = self.dataset.get_external_observation_space()
        observation_space['env_features'] = None

        return observation_space

    def set_dataset(self, dataset: TradingDataset):
        from yacht.data.renderers import TradingRenderer

        self.dataset = dataset
        self.prices = self.dataset.get_prices()

        # spaces
        self.observation_space = spaces.Dict(self.get_observation_space())
        assert self.observation_space['env_features'] is None or len(self.observation_space['env_features'].shape) == 1

        # episode
        self._end_tick = len(self.prices) - 2

        # Rendering
        self.renderer = TradingRenderer(
            data=self.prices,
            start=dataset.start,
            end=dataset.end
        )

    def reset(self):
        self._done = False
        self._tick_t = self._start_tick
        self._s_t = None
        self._a_t = None
        self._position_previous_t = None
        self._r_t = None
        self._total_value = 0.
        self.history = {}

        self.reward_schema.reset()

        return self.get_next_observation()

    def step(self, action: np.array):
        if self._done is True:
            info = self.on_done()

            return self._s_t, self._r_t, self._done, info
        else:
            # For a_t compute r_t
            action = self.action_schema.get_value(action)
            self._r_t = self.reward_schema.calculate_reward(
                action=action,
                dataset=self.dataset,
                current_index=self._tick_t - 1
            )

            # Log info for s_t
            info = self.create_info(action)
            self.update_history(info)

            # Update interval state after (s_t, a_t, r_t)
            self.update_total_value(action)

            # Get s_t+1
            self._tick_t += 1
            if self._tick_t == self._end_tick:
                self._done = True

            self._s_t = self.get_next_observation()

            if self._done is True:
                done_info = self.on_done()
                info.update(done_info)

            return self._s_t, self._r_t, self._done, info

    def create_info(self, action: np.array) -> dict:
        action = action.item()
        position = Position.build(position=action)

        info = dict(
            step=self._tick_t,
            done=self._done,
            action=action,
            position=position,
            reward=self._r_t,
            total_value=self._total_value,
            max_possible_value=(self._tick_t - self._start_tick) * self.action_schema.max_units_per_asset,
            total_value_completeness=round(self._total_value / self.max_possible_profit(stateless=True), 2)
        )

        return info

    def update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

            # There are no positions & actions before s_t.
            self.history['position'] = (self.window_size - 1) * [np.nan]
            self.history['action'] = (self.window_size - 1) * [0]

            current_date = self.dataset.index_to_datetime(self._tick_t)
            self.history['date'] = [
                current_date - datetime.timedelta(days=day_delta)
                for day_delta in range(1, self.window_size)
            ]

        # For easier processing add a position only when it is changing.
        position = info.pop('position')
        if self._position_previous_t != position:
            self.history['position'].append(position)
            self._position_previous_t = position
        else:
            self.history['position'].append(np.nan)

        current_date = self.dataset.index_to_datetime(self._tick_t)
        self.history['date'].append(current_date)

        for key, value in info.items():
            self.history[key].append(value)

    def get_next_observation(self) -> Dict[str, np.array]:
        observation = self.dataset[self._tick_t]

        return observation

    def on_done(self) -> dict:
        from yacht.evaluation import compute_backtest_results

        report = self.create_report()

        backtest_results, _ = compute_backtest_results(
            report,
            value_col_name='Total Value',
        )
        backtest_results = dict(zip(backtest_results.index, backtest_results.values))

        return backtest_results

    def create_report(self) -> pd.DataFrame:
        report = pd.DataFrame(
            data={
                'Date': self.history['date'],
                'Total Value': self.history['total_value']
            }
        )
        report['Date'] = pd.to_datetime(report['Date'])
        report.set_index('Date', inplace=True, drop=True)

        return report

    def create_baseline_report(self) -> pd.DataFrame:
        data = self.dataset.get_prices()

        report = pd.DataFrame(
            index=data.index,
            data={
                'Total Value': data.loc[:, 'Close']
            }
        )

        return report

    @abstractmethod
    def render(self, mode='human', name='trades.png'):
        pass

    def render_all(self, title, name='trades.png'):
        self.renderer.render(
            title=title,
            save_file_path=utils.build_graphics_path(self.dataset.storage_dir, name),
            positions=self.history['position'],
            actions=self.history['action']
        )

    def close(self):
        pass

    @abstractmethod
    def calculate_reward(self, action):
        pass

    @abstractmethod
    def update_total_value(self, action):
        pass

    @abstractmethod
    def max_possible_profit(self, stateless: bool = True) -> float:
        """
            @param stateless: If true will directly return the value, otherwise it will fill the 'history' for
                rendering and other observations.
        """
        pass
