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
