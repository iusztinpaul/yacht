from copy import copy
from typing import Optional, Dict

import numpy as np
from gym import spaces

from yacht.data.datasets import SampleAssetDataset
from yacht.environments import RewardSchema, ActionSchema
from yacht.environments.multi_asset import MultiAssetEnvironment


class OrderExecutionEnvironment(MultiAssetEnvironment):
    def __init__(
            self,
            name: str,
            dataset: SampleAssetDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            seed: int = 0,
            compute_metrics: bool = False,
            **kwargs
    ):
        super().__init__(name, dataset, reward_schema, action_schema, seed, compute_metrics, **kwargs)

        self.cash_used_on_last_tick = 0

    def _reset(self):
        super()._reset()

        self.cash_used_on_last_tick = 0

    def _get_observation_space(
            self,
            observation_space: Dict[str, Optional[spaces.Space]]
    ) -> Dict[str, Optional[spaces.Space]]:
        observation_space['env_features'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 2)
        )

        return observation_space

    def _get_next_observation(self, observation: Dict[str, np.array]) -> Dict[str, np.array]:
        assert self.is_history_initialized

        # The slicing method does not work for window_size == 1.
        if self.window_size > 1:
            past_used_positions = self.history['used_position'][-(self.window_size - 1):]
            past_used_time = self.history['used_time'][-(self.window_size - 1):]
        else:
            past_used_positions = []
            past_used_time = []

        used_positions = past_used_positions + [self._compute_used_position()]
        used_positions = np.array(used_positions, dtype=np.float32).reshape(-1, 1)
        used_time = past_used_time + [self._compute_used_time_ratio()]
        used_time = np.array(used_time, dtype=np.float32).reshape(-1, 1)

        observation['env_features'] = np.concatenate([used_positions, used_time], axis=-1)

        return observation

    def _initialize_history(self, history: dict) -> dict:
        history = super()._initialize_history(history)

        history['used_position'] = (self.window_size - 1) * [self._compute_used_position()]
        history['used_time'] = [
            self._compute_used_time_ratio(i) for i in range(self.window_size - 1)
        ]

        return history

    def update_internal_state(self, action: np.ndarray) -> dict:
        if self.is_done():
            # Make a copy to keep it for metrics.
            self.cash_used_on_last_tick = copy(self._total_cash)

            # Split money equally between the assets, if there is any cash position left for the current month.
            remaining_month_cash = np.tile(self._total_cash // self.dataset.num_assets, self.dataset.num_assets)
            if (remaining_month_cash > 0).all():
                # Update state action to preserve the history & statistics.
                self._a_t = remaining_month_cash / self._initial_cash_position
                changes = super().update_internal_state(self._a_t)
            else:
                self._a_t = np.zeros_like(action)
                changes = {
                    'total_cash': self._total_cash,
                    'total_units': np.copy(self._total_units.values)
                }

            changes['used_position'] = 1.
            changes['used_time'] = 1.
        else:
            changes = super().update_internal_state(action)
            changes['used_position'] = self._compute_used_position()
            changes['used_time'] = self._compute_used_time_ratio()

        return changes

    def _buy_asset(self, ticker: str, cash_ratio_to_use: float):
        if self._total_cash > 0:
            cash_to_use = abs(cash_ratio_to_use) * self._initial_cash_position
            cash_to_use = min(cash_to_use, self._total_cash)
            asset_price = self.dataset.get_decision_prices(self.t_tick, ticker).item()

            num_units_to_buy = cash_to_use / asset_price
            super()._buy_asset(ticker=ticker, num_units_to_buy=num_units_to_buy)

    def _filter_actions(self, actions: np.ndarray) -> np.ndarray:
        if self._total_cash <= 1.:
            actions = np.zeros_like(actions)

        return actions

    def _get_reward_schema_kwargs(self, next_state: Dict[str, np.ndarray]) -> dict:
        market_mean_price = self.dataset.compute_mean_price(
            self.dataset.sampled_dataset.start,
            self.dataset.sampled_dataset.end
        ).values
        next_price = self.dataset.get_decision_prices(self.t_tick).values

        return {
            'market_mean_price': market_mean_price,
            'next_price': next_price
        }

    def _compute_used_position(self) -> float:
        return (self._initial_cash_position - self._total_cash) / self._initial_cash_position

    def _compute_used_time_ratio(self, t_tick: Optional[int] = None) -> float:
        """
        Args:
            t_tick: The tick relative to the ratio will be computed. If `None` self.t_tick will be used.

        Returns:
            Computes the ratio from t_tick to the end of the month. If t_tick is at the very beginning of the month
            the function will return 0, otherwise if it is the last day of the month it will return 1.
        """

        if t_tick is None:
            t_tick = self.t_tick

        t_datetime = self.dataset.index_to_datetime(t_tick)
        start = self.dataset.sampled_dataset.start
        end = self.dataset.sampled_dataset.end

        # Decrement one day, because t_month_period.right is not included in the interval.
        # We want the data to have values between [0, 1], especially the 0 & 1 values.
        days_until_end_of_month = (end - t_datetime).days
        days_in_month = (end - start).days

        ratio = (days_in_month - days_until_end_of_month) / days_in_month
        assert 0 <= ratio <= 1, 't_tick / t_datetime it is not within the current month'

        return ratio

    def _on_done(self) -> dict:
        return {
            'cash_used_on_last_tick': self.cash_used_on_last_tick
        }

    def _is_done(self) -> bool:
        return False

    def _compute_render_all_graph_title(self, episode_metrics: dict) -> str:
        pa = round(episode_metrics['PA'], 4)
        cumulative_returns = round(episode_metrics['cumulative_returns'], 4)
        sharpe_ratio = round(episode_metrics['sharpe_ratio'], 4)
        max_drawdown = round(episode_metrics['max_drawdown'], 4)

        title = f'SR={sharpe_ratio};' \
                f'Cumulative Returns={cumulative_returns};' \
                f'PA={pa};' \
                f'Max Drawdown={max_drawdown}'

        return title
