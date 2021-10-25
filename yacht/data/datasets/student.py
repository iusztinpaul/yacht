import os
from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import Interval

from yacht import Mode
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
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        self.actions_store = pd.HDFStore(
            path=os.path.join(self.market.storage_dir, 'teacher_actions.h5'),
            mode='r'
        )

    def close(self):
        super().close()

        self.actions_store.close()

    def __getitem__(self, day_index: int) -> Dict[str, np.array]:
        data = super().__getitem__(day_index)
        data['action'] = self.actions_store.iloc[day_index].values

        return data
