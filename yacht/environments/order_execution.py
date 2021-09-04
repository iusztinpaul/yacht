from typing import Optional, Dict

import numpy as np
import pandas as pd
from gym import spaces

from yacht.data.datasets import ChooseAssetDataset
from yacht.environments import RewardSchema, ActionSchema
from yacht.environments.multi_asset import MultiAssetEnvironment


class OrderExecutionEnvironment(MultiAssetEnvironment):
    def __init__(
            self,
            name: str,
            dataset: ChooseAssetDataset,
            reward_schema: RewardSchema,
            action_schema: ActionSchema,
            seed: int = 0,
            render_on_done: bool = False,
            **kwargs
    ):
        super().__init__(name, dataset, reward_schema, action_schema, seed, render_on_done, **kwargs)

        # Compute month intervals for periodical order execution.
        self.include_weekends = kwargs.get('include_weekends', False)
        self.start = pd.Timestamp(dataset.start)
        self.end = pd.Timestamp(dataset.end)
        if self.include_weekends:
            freq = '1MS'
        else:
            freq = '1BMS'
        self.month_periods = list(pd.interval_range(start=self.start, end=self.end, freq=freq, closed='left'))
        assert len(self.month_periods) > 0
        if self.month_periods[0].left != self.start:
            self.month_periods.insert(
                0, pd.Interval(left=self.start, right=self.month_periods[0].left, closed='left')
            )
        if self.month_periods[-1].right != self.end:
            self.month_periods.append(
                pd.Interval(left=self.month_periods[-1].right, right=self.end, closed='both')
            )

        # Add more internal state variables.
        self.current_month_period_index = None
        self.monthly_cash_used = None

    def _reset(self):
        super()._reset()

        self.current_month_period_index = self._get_month_period_index()
        self.monthly_cash_used = 0

    def _get_month_period_index(self) -> int:
        """

        Returns: The month period index where the starting point falls into. Because we use a window logic the starting
            point will not fall every time in the first month period.
                E.g. The first interval is [2018-06-28, 2018-07-01) and we have a window_size of 10, then the starting
                point date will be 2018-07-07, which falls within the next period.

        """
        t_datetime = self.dataset.index_to_datetime(self.t_tick)
        for month_period_index, month_period in enumerate(self.month_periods):
            if t_datetime in month_period:
                return month_period_index

        raise RuntimeError('"self.t_tick outside the supported datetime range.')

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
            past_time_left = self.history['time_left'][-(self.window_size - 1):]
        else:
            past_used_positions = []
            past_time_left = []

        used_positions = past_used_positions + [self._compute_used_position()]
        used_positions = np.array(used_positions, dtype=np.float32).reshape(-1, 1)
        time_left = past_time_left + [self._compute_time_left_ratio()]
        time_left = np.array(time_left, dtype=np.float32).reshape(-1, 1)

        observation['env_features'] = np.concatenate([used_positions, time_left], axis=-1)

        return observation

    def _initialize_history(self, history: dict) -> dict:
        history = super()._initialize_history(history)

        history['used_position'] = (self.window_size - 1) * [self._compute_used_position()]
        history['time_left'] = [
            self._compute_time_left_ratio(i) for i in range(self.window_size - 1)
        ]

        return history

    def update_internal_state(self, action: np.ndarray) -> dict:
        if self._is_end_of_month_interval():
            # Split money equally between the assets, if there is any cash position left for the current month.
            remaining_month_cash = np.tile(self._total_cash // self.dataset.num_assets, self.dataset.num_assets)
            if (remaining_month_cash > 0).all():
                for ticker, ticker_cash_to_use in zip(self.dataset.asset_tickers, remaining_month_cash):
                    self._buy_asset(
                        ticker=ticker,
                        cash_ratio_to_use=self._total_cash / ticker_cash_to_use
                    )

            self.current_month_period_index += 1
            # This represents the monthly cash to be invested.
            self._total_cash = self._initial_cash_position

        changes = super().update_internal_state(action)
        changes['used_position'] = self._compute_used_position()
        changes['time_left'] = self._compute_time_left_ratio()

        return changes

    def _buy_asset(self, ticker: str, cash_ratio_to_use: float):
        cash_to_use = abs(cash_ratio_to_use) * self._initial_cash_position
        cash_to_use = min(cash_to_use, self._total_cash)
        asset_price = self.dataset.get_decision_prices(self.t_tick, ticker).item()

        if asset_price:
            num_units_to_buy = cash_to_use / asset_price
            super()._buy_asset(ticker=ticker, num_units_to_buy=num_units_to_buy)

    def _get_reward_function_kwargs(self, next_state: Dict[str, np.ndarray]) -> dict:
        current_month_interval = self.month_periods[self.current_month_period_index]
        market_mean_price = self.dataset.get_mean_over_period(current_month_interval.left, current_month_interval.right)

        return {
            'market_mean_price': market_mean_price,
            'next_price': self.dataset.get_decision_prices(self.t_tick)
        }

    def _is_done(self) -> bool:
        return len(self.month_periods) - 1 == self.current_month_period_index

    def _compute_used_position(self) -> float:
        return (self._initial_cash_position - self._total_cash) / self._initial_cash_position

    def _compute_time_left_ratio(self, t_tick: Optional[int] = None) -> float:
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
        t_month_period = None
        for month_period in self.month_periods:
            if t_datetime in month_period:
                t_month_period = month_period
                break

        assert t_month_period is not None, 't_datetime outside the supported datetime range.'

        days_until_end_of_month = (t_month_period.right - t_datetime).days
        days_in_month = (t_month_period.right - t_month_period.left).days

        ratio = (days_in_month - days_until_end_of_month) / days_in_month
        assert 0 <= ratio <= 1, 't_tick / t_datetime it is not within the current month'

        return ratio

    def _is_end_of_month_interval(self) -> bool:
        t_datetime = self.dataset.index_to_datetime(self.t_tick)
        t_month_interval = self.month_periods[self.current_month_period_index]

        return t_datetime == t_month_interval.right
