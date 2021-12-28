from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from pandas import Interval

from yacht import Mode
from yacht.data.datasets import DayFrequencyDataset, DatasetPeriod
from yacht.data.markets import Market
from yacht.data.scalers import Scaler
from yacht.data.transforms import Compose
from yacht.logger import Logger


class TeacherDayFrequencyDataset(DayFrequencyDataset):
    def __init__(
            self,
            ticker: str,
            market: Market,
            storage_dir: str,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            period: DatasetPeriod,
            render_intervals: List[Interval],
            render_tickers: List[str],
            mode: Mode,
            logger: Logger,
            scaler: Scaler,
            window_transforms: Optional[Compose] = None,
            window_size: int = 1,
            data: Dict[str, pd.DataFrame] = None
    ):
        super().__init__(
            ticker=ticker,
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
            scaler=scaler,
            window_transforms=window_transforms,
            window_size=window_size,
            data=data
        )

        self.cached_teacher_data = None

    def __getitem__(self, day_index: int) -> Dict[str, np.array]:
        """
        Args:
            day_index: The relative index the data will be given from.

        Returns:
            All the data within the [start, end] interval. Practically it will always return the same item.
        """

        if self.cached_teacher_data is None:
            day_features = self.data['1d'][self.features]
            day_features = self.scaler.transform(day_features)
            if self.window_transforms is not None:
                day_features = self.window_transforms(day_features)
            day_features = np.pad(
                day_features,
                ((0, self.window_size - day_features.shape[0]), (0, 0)),
                mode='edge'
            )
            day_features = np.expand_dims(day_features, axis=1)

            self.cached_teacher_data = day_features

        return {
            '1d': self.cached_teacher_data
        }
