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

        # General.
        self.dataset = dataset
        self.window_size = dataset.window_size
        self.prices = dataset.get_prices()
        self.reward_schema = reward_schema
        self.action_schema = action_schema

        # Spaces.
        self.action_space = self.action_schema.get_action_space()
        self.observation_space = spaces.Dict(self.get_observation_space())
        assert self.observation_space['env_features'] is None or len(self.observation_space['env_features'].shape) == 2

        # Ticks.
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 2  # Another -1 because the reward is calculated with one step further.
        self._tick_t = None

        # State.
        self._done = None
        self._s_t = None
        self._a_t = None
        self._position_previous_t = None
        self._r_t = None

        # Internal state.
        self._total_value = 0
        self._total_hits = 0
        self._total_profit_hits = 0
        self._total_loss_misses = 0
        self._num_longs = 0
        self._num_shorts = 0
        self._num_holds = 0

        # History.
        self.history = None

        # Rendering.
        self.renderer = TradingRenderer(
            data=self.prices,
            start=dataset.start,
            end=dataset.end
        )

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self._tick_t = self._start_tick

        # State.
        self._done = False
        self._s_t = None
        self._a_t = None
        self._position_previous_t = None
        self._r_t = None

        # Internal state.
        self._total_value = 0
        self._total_hits = 0
        self._total_profit_hits = 0
        self._total_loss_misses = 0
        self._num_longs = 0
        self._num_shorts = 0
        self._num_holds = 0

        # History.
        self.history = {}

        # Call custom code before computing the next observation.
        self._reset()

        self._s_t = self.get_next_observation()

        return self._s_t

    def _reset(self):
        pass

    def set_dataset(self, dataset: TradingDataset):
        # TODO: Find a better way to reinject the dataset.
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

    @property
    def current_profit(self):
        return self._total_value

    @property
    def intervals(self) -> List[str]:
        return self.dataset.intervals

    @property
    def is_history_initialized(self) -> bool:
        return len(self.history) > 0

    @property
    def observation_env_features_len(self) -> int:
        return self.observation_space['env_features'].shape[1] \
            if self.observation_space['env_features'] is not None \
            else 0

    def step(self, action: np.array):
        if self._done is True:
            episode_metrics = self.on_done()
            info = {
                'episode_metrics': episode_metrics
            }

            return self._s_t, self._r_t, self._done, info
        else:
            self._a_t = self.action_schema.get_value(action)

            # Update internal state after (s_t, a_t).
            self._tick_t += 1
            self.update_internal_state()

            # See if it is done after incrementing the tick & computing the internal state.
            self._done = self.is_done()

            # Log info for s_t.
            info = self.create_info()
            self.update_history(info)

            # Get next observation only to compute the reward. Do not update the env state yet.
            next_state = self.get_next_observation()

            # For a_t compute r_t.
            self._r_t = self.reward_schema.calculate_reward(
                action=self._a_t,
                current_state=self._s_t,
                next_state=next_state
            )

            if self._done is True:
                episode_metrics = self.on_done()
                info['episode_metrics'] = episode_metrics

            # Only at the end of the step update the env state.
            self._s_t = next_state

            return self._s_t, self._r_t, self._done, info

    @abstractmethod
    def update_internal_state(self):
        pass

    def get_observation_space(self) -> Dict[str, Optional[spaces.Space]]:
        observation_space = self.dataset.get_external_observation_space()
        observation_space['env_features'] = None
        observation_space = self._get_observation_space(observation_space)

        return observation_space

    def _get_observation_space(
            self,
            observation_space: Dict[str, Optional[spaces.Space]]
    ) -> Dict[str, Optional[spaces.Space]]:
        return observation_space

    def get_next_observation(self) -> Dict[str, np.array]:
        observation = self.dataset[self._tick_t]
        observation = self._get_next_observation(observation)

        return observation

    def _get_next_observation(self, observation: Dict[str, np.array]) -> Dict[str, np.array]:
        return observation

    def create_info(self) -> dict:
        action = self._a_t.item()
        position = Position.build(position=action)

        if position == Position.Long:
            self._num_longs += 1
        elif position == Position.Short:
            self._num_shorts += 1
        else:
            self._num_holds += 1

        # TODO: Find a better way to compute the hits ratios.
        # if self._r_t > 0:
        #     self._total_profit_hits += abs(action)
        #     self._total_hits += 1
        # else:
        #     self._total_loss_misses += abs(action)

        # if self._tick_t - self._start_tick != 0:
        #     hit_ratio = self._total_hits / (self._tick_t - self._start_tick)
        # else:
        #     hit_ratio = 0.

        info = dict(
            # State info
            step=self._tick_t - 1,  # Already incremented the tick at the start of the step() method.
            done=self._done,
            action=action,
            position=position,
            # Global information
            total_value=self._total_value,
            num_longs=self._num_longs,
            num_shorts=self._num_shorts,
            num_holds=self._num_holds,
            # profit_hits=self._total_profit_hits,
            # loss_misses=self._total_loss_misses,
            # hit_ratio=hit_ratio
        )

        info = self._create_info(info)

        return info

    def _create_info(self, info: dict) -> dict:
        return info

    def update_history(self, info):
        if not self.history:
            self.history = {
                key: (self.window_size - 1) * [np.nan] for key in info.keys()
            }

            self.history['action'] = (self.window_size - 1) * [0]
            self.history['total_value'] = (self.window_size - 1) * [self._total_value]

            # Map indices to their corresponding dates.
            current_date = self.dataset.index_to_datetime(self._tick_t)
            self.history['date'] = [
                current_date - datetime.timedelta(days=day_delta)
                for day_delta in range(1, self.window_size)
            ]

            self.history = self._initialize_history(self.history)

        # For easier processing add a position only when it is changing.
        position = info.pop('position')
        if self._position_previous_t != position:
            self.history['position'].append(position)
            self._position_previous_t = position
        else:
            self.history['position'].append(np.nan)

        # Map index_t to its corresponding date.
        current_date = self.dataset.index_to_datetime(self._tick_t)
        self.history['date'].append(current_date)

        for key, value in info.items():
            self.history[key].append(value)

        self.history = self._update_history(self.history)

    def _initialize_history(self, history: dict) -> dict:
        return history

    def _update_history(self, history: dict) -> dict:
        return history

    def is_done(self) -> bool:
        is_end_sequence = self._tick_t == self._end_tick

        return is_end_sequence or self._is_done()

    def _is_done(self) -> bool:
        return False

    def on_done(self) -> dict:
        """
            Returns episode metrics in a dictionary format.
        """
        from yacht.evaluation import compute_backtest_results

        report = self.create_report()

        backtest_results, _ = compute_backtest_results(
            report,
            value_col_name='total',
        )

        backtest_results = dict(
            zip(backtest_results.index, backtest_results.values)
        )
        # Map all indices from plain english title to snake case for consistency.
        snake_case_backtest_results = dict()
        for k, v in backtest_results.items():
            snake_case_backtest_results[utils.english_title_to_snake_case(k)] = v

        return snake_case_backtest_results

    def create_report(self) -> pd.DataFrame:
        if 'total_assets' in self.history:
            data = {
                'date': self.history['date'],
                'total': self.history['total_assets']
            }
        else:
            data = {
                'date': self.history['date'],
                'total': self.history['total_value']
            }
        report = pd.DataFrame(data=data)
        report['date'] = pd.to_datetime(report['date'])
        report.set_index('date', inplace=True, drop=True)

        return report

    def create_baseline_report(self) -> pd.DataFrame:
        data = self.dataset.get_prices()

        report = pd.DataFrame(
            index=data.index,
            data={
                'total': data.loc[:, 'Close']
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
    def max_possible_profit(self, stateless: bool = True) -> float:
        """
            @param stateless: If true will directly return the value, otherwise it will fill the 'history' for
                rendering and other observations.
        """
        pass
