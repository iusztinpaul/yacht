import datetime
import time
from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple

import gym
from gym import spaces
import numpy as np
from pandas._libs.tslibs.offsets import BDay
from stable_baselines3.common.utils import set_random_seed

from yacht import utils
from yacht.data.datasets import SampleAssetDataset
from yacht.environments.action_schemas import ActionSchema
from yacht.environments.reward_schemas import RewardSchema


class BaseAssetEnvironment(gym.Env, ABC):
    def __init__(
            self,
            name: str,
            dataset: SampleAssetDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            compute_metrics: bool = False
    ):
        # Environment name
        self.name = name

        # Environment general requirements.
        self.dataset = dataset
        self.window_size = dataset.past_window_size
        self.reward_schema = reward_schema
        self.action_schema = action_schema

        # Spaces.
        self.action_space = self.action_schema.get_action_space()
        self.observation_space = spaces.Dict(self.get_observation_space())
        assert self.observation_space['env_features'] is None or len(self.observation_space['env_features'].shape) == 2

        # Ticks.
        self.start_tick = self.dataset.first_observation_index
        self.end_tick = self.dataset.last_observation_index
        self.t_tick = None

        # MDP state.
        self._done = None
        self._s_t = None
        self._a_t = None
        self._r_t = None

        # Internal state.
        self._total_cash = 0
        self._total_hits = 0
        self._total_profit_hits = 0
        self._total_loss_misses = 0
        self._num_longs = 0
        self._num_shorts = 0
        self._num_holds = 0

        # History.
        self.history = None

        # Rendering.
        self.renderer = self.build_renderer()

        # Metrics
        self.compute_metrics = compute_metrics

        # Time Metrics
        self.data_time_length = np.nan
        self.env_time_length = np.nan
        self.env_time_end = np.nan
        self.agent_time_length = np.nan

    def seed(self, seed=None):
        set_random_seed(seed=seed, using_cuda=True)

    def reset(self):
        # Choose a random ticker for every instance of the environment.
        self.dataset.sample()

        # Rendering.
        self.renderer = self.build_renderer()

        # Actions.
        self.action_schema.reset()

        # Ticks.
        self.start_tick = self.dataset.first_observation_index
        self.end_tick = self.dataset.last_observation_index
        self.t_tick = self.start_tick

        # MDP state.
        self._done = False
        self._s_t = None
        self._a_t = None
        self._r_t = None

        # Internal state.
        self._total_cash = 0
        self._total_hits = 0
        self._total_profit_hits = 0
        self._total_loss_misses = 0
        self._num_longs = 0
        self._num_shorts = 0
        self._num_holds = 0

        # Call custom code before computing the next observation.
        self._reset()

        # History.
        self.history = self.initialize_history()

        # Time Metrics
        self.data_time_length = np.nan
        self.env_time_length = np.nan
        self.env_time_end = np.nan
        self.agent_time_length = np.nan

        self._s_t = self.get_observation()

        # Compute env_time_end here so we are able to calculate the agent time from the first step.
        self.env_time_end = time.time()

        return self._s_t

    def _reset(self):
        pass

    def build_renderer(self):
        from yacht.data.renderers import AssetEnvironmentRenderer

        renderer = AssetEnvironmentRenderer(
            data=self.dataset.get_prices(),
            start=self.dataset.sampled_dataset.start,
            end=self.dataset.sampled_dataset.end,
            unadjusted_start=self.dataset.sampled_dataset.unadjusted_start
        )

        return renderer

    @property
    def is_history_initialized(self) -> bool:
        return self.history is not None

    @property
    def intervals(self) -> List[str]:
        return self.dataset.intervals

    @property
    def observation_env_features_len(self) -> int:
        return self.observation_space['env_features'].shape[1] \
            if self.observation_space['env_features'] is not None \
            else 0

    def step(self, action: np.ndarray):
        # The agent time is computed between the last time it called the environment &
        # the start of the environment logic.
        if np.isfinite(self.env_time_end):
            self.agent_time_length = time.time() - self.env_time_end
        env_start_time = time.time()

        if self._done is True:
            episode_metrics, report = self.on_done()
            info = {
                'episode_metrics': episode_metrics,
                'report': report,
                'ticker': self.dataset.asset_tickers
            }

            return self._s_t, self._r_t, self._done, info
        else:
            self.t_tick += 1

            self._a_t = self.action_schema.get_value(action)
            self._a_t = self._filter_actions(self._a_t)

            # Update internal state after (s_t, a_t).
            changes = self.update_internal_state(self._a_t)
            changes['action'] = self._a_t
            # Update history with the internal changes so the next observation can be computed.
            self.update_history(changes)

            # Get next observation only to compute the reward. Do not update the env state yet.
            # The next observation should be computed only after updating the internal state to the history.
            next_state = self.get_observation()

            # For a_t compute r_t.
            reward_schema_kwargs = self._get_reward_schema_kwargs(next_state)
            self._r_t = self.reward_schema.calculate_reward(
                action=self._a_t,
                **reward_schema_kwargs
            )

            # See if it is done after incrementing the tick & computing the internal state.
            self._done = self.is_done()

            # The env_time_length variable does not contain the time to compute the metrics. It is not that important
            # because the metrics are computed only at backtest.
            self.env_time_end = time.time()
            self.env_time_length = self.env_time_end - env_start_time

            # Log info for s_t.
            info = self.create_info()
            # Add the rest of the step information to the history.
            self.update_history(info)

            if self._done is True:
                episode_metrics, report = self.on_done()
                info['episode_metrics'] = episode_metrics
                info['report'] = report
                info['ticker'] = self.dataset.asset_tickers

            # Only at the end of the step update the env state.
            self._s_t = next_state

            return self._s_t, self._r_t, self._done, info

    @abstractmethod
    def update_internal_state(self, action: np.ndarray) -> dict:
        """
            Returns: The internal state variables that has to be updated before creating the next observation.
        """
        pass

    def _filter_actions(self, actions: np.ndarray) -> np.ndarray:
        return actions

    def _get_reward_schema_kwargs(self, next_state: Dict[str, np.ndarray]) -> dict:
        return {
            'current_state': self.inverse_scaling(self._s_t),
            'next_state': self.inverse_scaling(next_state)
        }

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

    def get_observation(self) -> Dict[str, np.ndarray]:
        start_data_time = time.time()

        observation = self.dataset[self.t_tick]
        # Replace the value of 'env_features' if you want to add custom data.
        # observation['env_features'] = ...

        observation = self._get_env_observation(observation)
        observation = self.scale_env_observation(observation)

        self.data_time_length = time.time() - start_data_time

        return observation

    def _get_env_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return observation

    def scale_env_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # External / Dataset observations are already scaled. Here you should scale only env dependent data.
        return observation

    def inverse_scaling(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation = self.dataset.inverse_scaling(observation)
        observation = self.inverse_scale_env_observation(observation)

        return observation

    def inverse_scale_env_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return observation

    def create_info(self) -> dict:
        if self._a_t is not None:
            action = self._a_t
            position = self.map_to_position(action)
        else:
            action = np.array([])
            position = np.array([])

        if position.size != 0:
            for p in position:
                if p == 1:
                    self._num_longs += 1
                elif p == -1:
                    self._num_shorts += 1
                else:
                    self._num_holds += 1

        info = dict(
            # MDP information.
            step=self.t_tick - self.window_size,
            action=action,
            reward=self._r_t,
            # Interval state information.
            total_cash=self._total_cash,
            num_longs=self._num_longs,
            num_shorts=self._num_shorts,
            num_holds=self._num_holds,
            data_time_length=self.data_time_length,
            env_time_length=self.env_time_length,
            agent_time_length=self.agent_time_length,
        )

        info = self._create_info(info)

        return info

    def _create_info(self, info: dict) -> dict:
        return info

    @classmethod
    def map_to_position(cls, action: np.ndarray) -> np.ndarray:
        return np.sign(action)

    def update_history(self, changes: dict):
        # Update date at any first call for the current self._tick_t
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
        def _get_day_offset(day_delta):
            if self.dataset.sampled_dataset.include_weekends:
                return datetime.timedelta(days=day_delta)
            else:
                return BDay(day_delta)
        history = dict()

        history_keys = set(self.create_info().keys())
        history_keys.add('date')
        for key in history_keys:
            if key == 'action':
                history['action'] = self.window_size * [[0] * self.dataset.num_assets]
            elif key == 'total_cash':
                history['total_cash'] = self.window_size * [self._total_cash]
            elif key == 'date':
                current_date = self.dataset.index_to_datetime(self.window_size)
                history['date'] = [
                    current_date - _get_day_offset(day_delta)
                    for day_delta in range(self.window_size, 0, -1)
                ]
            else:
                history[key] = self.window_size * [np.nan]

        # Initialize custom states.
        history = self._initialize_history(history)

        return history

    def _initialize_history(self, history: dict) -> dict:
        # The total quantity of your assets.
        history['total_units'] = []
        # Parameter where to store the total value of your assets = units * price.
        history['total_assets'] = []

        return history

    def _should_update_history(self, key: str, changes: dict):
        """
            Because the history can be updated on multiple calls on the step function check if in the changes
            dictionary there is the desired key & that it was not already added in the current step.
        """
        return key in changes and len(self.history.get(key, dict())) <= self.t_tick

    def _update_history(self, history: dict) -> dict:
        return history

    def is_done(self) -> bool:
        is_end_sequence = self.t_tick >= self.end_tick

        return is_end_sequence or self._is_done()

    def _is_done(self) -> bool:
        return False

    def on_done(self) -> Tuple[dict, dict]:
        """
            Returns episode metrics in a dictionary format &
                The history report in a DataFrame
        """
        from yacht.evaluation import compute_backtest_metrics

        if self.compute_metrics:
            report = self.create_report()

            episode_metrics, _ = compute_backtest_metrics(
                report,
                total_assets_col_name='total_assets',
            )
            episode_metrics.update(
                self._on_done(report)
            )

            if self.dataset.should_render:
                self.render_all(
                    report=report,
                    title=self._compute_render_all_graph_title(episode_metrics),
                    name=f'{self.name}.png'
                )

            return episode_metrics, report

        return self._on_done(), dict()

    def _on_done(self, report: Optional[dict] = None) -> dict:
        return dict()

    def _compute_render_all_graph_title(self, episode_metrics: dict) -> str:
        annual_return = round(episode_metrics['annual_return'], 4)
        cumulative_returns = round(episode_metrics['cumulative_returns'], 4)
        sharpe_ratio = round(episode_metrics['sharpe_ratio'], 4)
        max_drawdown = round(episode_metrics['max_drawdown'], 4)

        title = f'SR={sharpe_ratio};' \
                f'Cumulative Returns={cumulative_returns};' \
                f'Annual Return={annual_return};' \
                f'Max Drawdown={max_drawdown}'

        return title

    def create_report(self) -> Dict[str, Union[np.ndarray, list]]:
        # The whole range it is computed only for rendering purposes.
        start = self.dataset.sampled_dataset.start
        end = self.dataset.sampled_dataset.end
        # We really care only about the actions taken in the unadjusted range.
        unadjusted_start = self.dataset.sampled_dataset.unadjusted_start

        dates = utils.compute_period_range(start, end, self.dataset.include_weekends)
        unadjusted_dates = utils.compute_period_range(unadjusted_start, end, self.dataset.include_weekends)

        prices = self.dataset.get_decision_prices()
        unadjusted_prices = prices.loc[unadjusted_dates]
        prices = prices.loc[dates]

        data = {
            'date': dates,
            'price': prices.values,
            'action': np.array(self.history['action'], dtype=np.float32),
            'longs': np.array(self.history['num_longs']),
            'shorts': np.array(self.history['num_shorts']),
            'holds': np.array(self.history['num_holds']),
            'total_cash': np.array(self.history['total_cash'], dtype=np.float64)
        }
        if len(self.history['total_units']) > 0:
            data['total_units'] = np.array(self.history['total_units'], dtype=np.float32)
        if len(self.history['total_assets']) > 0:
            data['total_assets'] = np.array(self.history['total_assets'], dtype=np.float64)
            
        data = self.pad_report(data)

        data['unadjusted_dates'] = unadjusted_dates
        data['unadjusted_prices'] = unadjusted_prices.values
        data['unadjusted_actions'] = data['action'][self.window_size:]

        return data
    
    def pad_report(self, report: Dict[str, Union[np.ndarray, list]]) -> Dict[str, Union[np.ndarray, list]]:
        edge_keys = ('total_cash', 'total_units', 'total_assets')
        zero_keys = ('action', 'longs', 'shorts', 'holds')

        remaining_period_length = self.end_tick - self.t_tick
        padding_axis_one_dim = (0, remaining_period_length)
        padding_axis_two_dims = ((0, remaining_period_length), (0, 0))
        for k, v in report.items():
            if isinstance(v, np.ndarray) and len(v.shape) == 1:
                padding_axis = padding_axis_one_dim
            else:
                padding_axis = padding_axis_two_dims

            if k in edge_keys:
                report[k] = np.pad(report[k], padding_axis, mode='edge')
            elif k in zero_keys:
                report[k] = np.pad(report[k], padding_axis, mode='constant', constant_values=(0, ))

        return report

    @abstractmethod
    def render(self, mode='human', name='trades.png'):
        pass

    def render_all(self, report: dict, title: str, name='trades.png'):
        renderer_kwargs = {
            'title': title,
            'save_file_path': self._build_render_path(name),
            'tickers': self.dataset.asset_tickers,
            'actions': report['action'],
            'total_cash': report['total_cash']
        }
        if len(self.history['total_assets']) > 0:
            renderer_kwargs['total_assets'] = report['total_assets']
        if len(self.history['total_units']) > 0:
            renderer_kwargs['total_units'] = report['total_units']

        self.renderer.render(**renderer_kwargs)
        
    def _build_render_path(self, name: str) -> str:
        if not name.endswith('.png'):
            name = f'{name}.png'
            
        return utils.build_graphics_path(self.dataset.storage_dir, name)

    def close(self):
        pass
