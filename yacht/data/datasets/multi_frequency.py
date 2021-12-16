from collections import defaultdict
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
from gym import spaces
from pandas import Interval

from yacht import Mode
from yacht.data.datasets import SingleAssetDataset, DatasetPeriod
from yacht.data.markets import Market
from yacht.data.scalers import Scaler
from yacht.data.transforms import Compose
from yacht.logger import Logger


class DayMultiFrequencyDataset(SingleAssetDataset):
    INTERVAL_TO_BARS_PER_DAY = {
        # All hours ( crypto).
        True: {
            '1d': 1,
            '1h': 24
        },
        # Trading hours.
        False: {
            '1d': 1,
            '1h': 7
        }
    }

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

        assert set(intervals).issubset(set(self.supported_intervals)), 'Requested intervals are not supported.'

    def __len__(self):
        return len(self.data['1d'])

    def get_external_observation_space(self) -> Dict[str, spaces.Space]:
        observation_space = dict()
        for interval in self.intervals:
            interval_bars = self.INTERVAL_TO_BARS_PER_DAY[self.include_weekends][interval]
            observation_space[interval] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, interval_bars, len(self.features)),
                dtype=np.float32
            )

        return observation_space

    @classmethod
    def get_day_bar_units_for(cls, interval: str, include_weekends: bool) -> int:
        return cls.INTERVAL_TO_BARS_PER_DAY[include_weekends][interval]

    @property
    def supported_intervals(self):
        return list(self.INTERVAL_TO_BARS_PER_DAY[self.include_weekends].keys())

    def __getitem__(self, day_index: int) -> Dict[str, np.array]:
        """
        Args:
            day_index: The relative index the data will be given from.

        Returns:
            The data features within the [day_index - window_size + 1, day_index] interval.
        """

        window_item: Dict[str, np.ndarray] = dict()
        for interval in self.intervals:
            features = self.data[interval][self.features]
            bars_per_day = self.INTERVAL_TO_BARS_PER_DAY[self.include_weekends][interval]
            start_index = (day_index - self.window_size + 1) * bars_per_day
            end_index = (day_index + 1) * bars_per_day

            features = features[start_index: end_index]
            features = self.scaler.transform(features)
            features = features.reshape(self.window_size, bars_per_day, len(self.features))
            if self.window_transforms is not None:
                features = self.window_transforms(features)

            window_item[interval] = features

        return window_item
