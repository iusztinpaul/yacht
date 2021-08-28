import numpy as np
import pandas as pd

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
        self.month_intervals = list(pd.interval_range(start=self.start, end=self.end, freq=freq))
        if self.month_intervals[0].left != self.start:
            self.month_intervals.insert(
                0, pd.Interval(left=self.start, right=self.month_intervals[0].left)
            )
        if self.month_intervals[-1].right != self.end:
            self.month_intervals.append(
                pd.Interval(left=self.month_intervals[-1].right, right=self.end)
            )

        # Add more internal state variables.
        self.current_month_interval_index = 0
        self.monthly_cash_used = 0

    def _reset(self):
        super()._reset()

        self.current_month_interval_index = 0
        self.monthly_cash_used = 0

    def update_internal_state(self, action: np.ndarray) -> dict:
        if self.is_end_of_month_interval():
            self.current_month_interval_index += 1
            # This represents the monthly cash to be invested.
            self._total_cash = self._initial_cash_position

            # TODO: Buy with the remaining cash position.

        return {}

    def is_end_of_month_interval(self) -> bool:
        t_datetime = self.dataset.index_to_datetime(self.t_tick)
        t_month_interval = self.month_intervals[self.current_month_interval_index]

        return t_datetime == t_month_interval.right

    def _is_done(self) -> bool:
        return super()._is_done() or len(self.month_intervals) - 1 == self.current_month_interval_index
