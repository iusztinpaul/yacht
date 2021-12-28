import os
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import Interval, HDFStore

from yacht import Mode, utils
from yacht.data.datasets import MultiAssetDataset, SingleAssetDataset, DatasetPeriod
from yacht.data.markets import Market
from yacht.logger import Logger


class StudentMultiAssetDataset(MultiAssetDataset):
    def __init__(
            self,
            datasets: List[SingleAssetDataset],
            storage_dir: str,
            market: Market,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            period: DatasetPeriod,
            render_intervals: List[Interval],
            render_tickers: List[str],
            mode: Mode,
            logger: Logger,
            window_size: int = 1
    ):
        super().__init__(
            datasets=datasets,
            market=market,
            storage_dir=storage_dir,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            period=period,
            render_intervals=render_intervals,
            render_tickers=render_tickers,
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        teacher_actions_path = os.path.join(self.market.storage_dir, 'teacher_actions.h5')
        actions_store = pd.HDFStore(
            path=teacher_actions_path,
            mode='r'
        )
        self.actions = self.get_teacher_actions(actions_store)
        actions_store.close()

    def get_teacher_actions(self, actions_store: HDFStore) -> pd.DataFrame:
        from yacht.environments.order_execution import ExportTeacherActionsOrderExecutionEnvironment

        teacher_actions = actions_store[ExportTeacherActionsOrderExecutionEnvironment.create_key(self)]
        # Firstly, create a template with the desired dates in case any actions are missing within the past window_size.
        start = self.datasets[0].start
        end = self.datasets[0].end
        include_weekends = self.datasets[0].include_weekends
        index = utils.compute_period_range(start, end, include_weekends)
        template_actions = pd.DataFrame(index=index, columns=teacher_actions.columns)
        # Update the template with the known teacher values within the dataset [start:end] interval.
        teacher_actions = teacher_actions.loc[start:end]
        template_actions.update(teacher_actions)
        template_actions = template_actions.fillna(0)

        return template_actions

    def __getitem__(self, day_index: int) -> Dict[str, np.array]:
        data = super().__getitem__(day_index)
        # For data within window [t - window_size + 1; t] the action is taken at t + 1.
        action_index = day_index + 1
        if action_index < len(self.actions):
            data['teacher_action'] = self.actions.iloc[day_index + 1].values
        else:
            # There is no action for the final window observation.
            data['teacher_action'] = -1

        return data
