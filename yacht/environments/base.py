import datetime
import logging
from abc import abstractmethod, ABC
from copy import copy
from typing import Dict, List, Optional, Union, Tuple

import gym
import pandas as pd
from gym import spaces
import numpy as np
from stable_baselines3.common.utils import set_random_seed

from yacht import utils
from yacht.data.datasets import ChooseAssetDataset
from yacht.environments.action_schemas import ActionSchema
from yacht.environments.enums import Position
from yacht.environments.reward_schemas import RewardSchema

logger = logging.getLogger(__file__)


class BaseAssetEnvironment(gym.Env, ABC):
    def __init__(
            self,
            name: str,
            dataset: ChooseAssetDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            seed: int = 0,
            render_on_done: bool = False
    ):
        from yacht.data.renderers import TradingRenderer

        # Environment name
        self.name = name
        self.given_seed = seed

        # Environment general requirements.
        self.dataset = copy(dataset)
        self.window_size = dataset.window_size
        self.prices = dataset.get_prices()
        self.reward_schema = reward_schema
        self.action_schema = action_schema

        # Spaces.
        self.action_space = self.action_schema.get_action_space()
        self.observation_space = spaces.Dict(self.get_observation_space())
        assert self.observation_space['env_features'] is None or len(self.observation_space['env_features'].shape) == 2

        # Ticks.
        self.start_tick = self.window_size - 1
        self.end_tick = len(self.prices) - 2  # Another -1 because the reward is calculated with one step further.
        self.t_tick = None

        # State.
        self._done = None
        self._s_t = None
        self._a_t = None
        self._position_previous_t = None
        self._r_t = None

        # Internal state.
        self.total_value = 0
        self._total_hits = 0
        self._total_profit_hits = 0
        self._total_loss_misses = 0
        self._num_longs = 0
        self._num_shorts = 0
        self._num_holds = 0

        # History.
        self.history = None
        self.is_history_initialized = False

        # Rendering.
        self.renderer = TradingRenderer(
            data=self.prices,
            start=dataset.start,
            end=dataset.end
        )
        self.render_on_done = render_on_done

    def seed(self, seed=None):
        self.given_seed = seed

        set_random_seed(seed=seed, using_cuda=True)

    def reset(self):
        # Choose a random ticker for every instance of the environment.
        self.dataset.choose_ticker()

        self.t_tick = self.start_tick

        # State.
        self._done = False
        self._s_t = None
        self._a_t = None
        self._position_previous_t = None
        self._r_t = None

        # Internal state.
        self.total_value = 0
        self._total_hits = 0
        self._total_profit_hits = 0
        self._total_loss_misses = 0
        self._num_longs = 0
        self._num_shorts = 0
        self._num_holds = 0

        # History.
        self.history = self.initialize_history()
        self.is_history_initialized = True

        # Call custom code before computing the next observation.
        self._reset()

        self._s_t = self.get_next_observation()

        return self._s_t

    def _reset(self):
        pass

    def set_dataset(self, dataset: ChooseAssetDataset):
        # TODO: Find a better way to reinject the dataset.
        from yacht.data.renderers import TradingRenderer

        self.dataset = dataset
        self.prices = self.dataset.get_prices()

        # spaces
        self.observation_space = spaces.Dict(self.get_observation_space())
        assert self.observation_space['env_features'] is None or len(self.observation_space['env_features'].shape) == 1

        # episode
        self.end_tick = len(self.prices) - 2

        # Rendering
        self.renderer = TradingRenderer(
            data=self.prices,
            start=dataset.start,
            end=dataset.end
        )

    @property
    def current_ticker(self) -> str:
        return self.dataset.current_ticker

    @property
    def intervals(self) -> List[str]:
        return self.dataset.intervals

    @property
    def observation_env_features_len(self) -> int:
        return self.observation_space['env_features'].shape[1] \
            if self.observation_space['env_features'] is not None \
            else 0

    def step(self, action: np.array):
        if self._done is True:
            episode_metrics, report = self.on_done()
            info = {
                'episode_metrics': episode_metrics,
                'report': report,
                'ticker': self.current_ticker
            }

            return self._s_t, self._r_t, self._done, info
        else:
            self._a_t = self.action_schema.get_value(action)

            # Update internal state after (s_t, a_t).
            self.t_tick += 1
            changes = self.update_internal_state(self._a_t.item())
            # Update history with the internal changes so the next observation can be computed.
            self.update_history(changes)

            # Get next observation only to compute the reward. Do not update the env state yet.
            next_state = self.get_next_observation()

            # For a_t compute r_t.
            self._r_t = self.reward_schema.calculate_reward(
                action=self._a_t,
                current_state=self._s_t,
                next_state=next_state
            )

            # See if it is done after incrementing the tick & computing the internal state.
            self._done = self.is_done()

            # Log info for s_t.
            info = self.create_info()
            # Add the rest of the step information to the history.
            self.update_history(info)

            if self._done is True:
                episode_metrics, report = self.on_done()
                info['episode_metrics'] = episode_metrics
                info['report'] = report
                info['ticker'] = self.current_ticker

            # Only at the end of the step update the env state.
            self._s_t = next_state

            return self._s_t, self._r_t, self._done, info

    @abstractmethod
    def update_internal_state(self, action: Union[int, float]) -> dict:
        """
            Returns: The internal state variables that were updated.
        """
        pass

    def get_observation_space(self) -> Dict[str, Optional[spaces.Space]]:
        observation_space = self.dataset.get_external_observation_space()
        # Add 'env_features' observation space for custom data
        # observation_space['env_features'] = spaces.Box(low=-np.inf, high=np.inf, shape=(0, ), dtype=np.float32)

        observation_space = self._get_observation_space(observation_space)

        return observation_space

    def _get_observation_space(
            self,
            observation_space: Dict[str, Optional[spaces.Space]]
    ) -> Dict[str, Optional[spaces.Space]]:
        return observation_space

    def get_next_observation(self) -> Dict[str, np.array]:
        observation = self.dataset[self.t_tick]
        # Replace the value of 'env_features' if you want to add custom data.
        # observation['env_features'] = ...

        observation = self._get_next_observation(observation)

        return observation

    def _get_next_observation(self, observation: Dict[str, np.array]) -> Dict[str, np.array]:
        return observation

    def create_info(self) -> dict:
        if self._a_t is not None:
            action = self._a_t.item()
            position = Position.build(position=action)
        else:
            action = None
            position = None

        if position == Position.Long:
            self._num_longs += 1
        elif position == Position.Short:
            self._num_shorts += 1
        else:
            self._num_holds += 1

        # TODO: Find a better way to compute the hits ratios.
        if self._r_t is not None:
            if self._r_t > 0:
                self._total_profit_hits += abs(action)
                self._total_hits += 1
            else:
                self._total_loss_misses += abs(action)

            if self.t_tick - self.start_tick != 0:
                hit_ratio = self._total_hits / (self.t_tick - self.start_tick)
            else:
                hit_ratio = 0.
        else:
            hit_ratio = 0.

        info = dict(
            # MDP information.
            step=self.t_tick - 1,  # Already incremented the tick at the start of the step() method.
            done=self._done,
            action=action,
            position=position,
            reward=self._r_t,
            # Interval state information.
            total_value=self.total_value,
            num_longs=self._num_longs,
            num_shorts=self._num_shorts,
            num_holds=self._num_holds,
            profit_hits=self._total_profit_hits,
            loss_misses=self._total_loss_misses,
            hit_ratio=hit_ratio
        )

        info = self._create_info(info)

        return info

    def _create_info(self, info: dict) -> dict:
        return info

    def update_history(self, changes: dict):
        if self._should_update_history(key='position', changes=changes):
            # For easier processing add a position only when it is changing.
            position = changes.pop('position')
            if self._position_previous_t != position:
                self.history['position'].append(position)
                self._position_previous_t = position
            else:
                self.history['position'].append(np.nan)

        # Update date at any time for the current self._tick_t
        changes['date'] = None
        if self._should_update_history(key='date', changes=changes):
            # Map index_t to its corresponding date.
            current_date = self.dataset.index_to_datetime(self.t_tick)
            self.history['date'].append(current_date)

        for key, value in changes.items():
            if self._should_update_history(key=key, changes=changes):
                self.history[key].append(value)

        self.history = self._update_history(self.history)

    def initialize_history(self):
        history = dict()
        history_keys = set(self.create_info().keys())
        history_keys.add('date')
        for key in history_keys:
            if key == 'action':
                history['action'] = (self.window_size - 1) * [0]
            elif key == 'total_value':
                history['total_value'] = (self.window_size - 1) * [self.total_value]
            elif key == 'date':
                current_date = self.dataset.index_to_datetime(self.t_tick)
                history['date'] = [
                    current_date - datetime.timedelta(days=day_delta)
                    for day_delta in range(1, self.window_size)
                ]
            else:
                history[key] = (self.window_size - 1) * [np.nan]

        # Initialize custom states.
        history = self._initialize_history(history)

        return history

    def _initialize_history(self, history: dict) -> dict:
        return history

    def _should_update_history(self, key: str, changes: dict):
        """
            Because the history can be updated on multiple calls on the step function check if in the changes
            dictionary there is the desired key & that it was not already added in the current step.
        """
        return key in changes and len(self.history.get(key, dict())) < self.t_tick

    def _update_history(self, history: dict) -> dict:
        return history

    def is_done(self) -> bool:
        is_end_sequence = self.t_tick == self.end_tick

        return is_end_sequence or self._is_done()

    def _is_done(self) -> bool:
        return False

    def on_done(self) -> Tuple[dict, pd.DataFrame]:
        """
            Returns episode metrics in a dictionary format.
        """
        from yacht.evaluation import compute_backtest_metrics

        report = self.create_report()

        episode_metrics, _ = compute_backtest_metrics(
            report,
            value_col_name='total',
        )

        if self.render_on_done:
            sharpe_ratio = round(episode_metrics['sharpe_ratio'], 4)
            total_assets = round(self.history['total_assets'][-1], 4)
            annual_return = round(episode_metrics['annual_return'], 4)

            title = f'SR={sharpe_ratio};' \
                    f'Total Assets={total_assets};' \
                    f'Annual Return={annual_return}'
            self.render_all(title=title, name=f'{self.name}.png')

        return episode_metrics, report

    def create_report(self) -> pd.DataFrame:
        data = {
            'date': self.history['date'],
            'action': self.history['action'],
            'price': self.prices.loc[:, 'Close'][:self.end_tick]
        }
        if 'total_assets' in self.history:
            data['total'] = self.history['total_assets']
        else:
            data['total'] = self.history['total_value']

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
        name = f'{self.current_ticker}_{self.given_seed}_{name}'
        if not name.endswith('.png'):
            name = f'{name}.png'

        self.renderer.render(
            title=title,
            save_file_path=utils.build_graphics_path(self.dataset.storage_dir, name),
            positions=self.history['position'],
            actions=self.history['action'],
            total_value=self.history['total_value']
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
