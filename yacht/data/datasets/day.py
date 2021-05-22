import logging
from datetime import datetime
from typing import List

import numpy as np

from yacht.data.datasets import TradingDataset, TrainValMixin
from yacht.data.markets import Market
from yacht.data.normalizers import Normalizer

logger = logging.getLogger(__file__)


class DayForecastDataset(TradingDataset):
    INTERVAL_TO_DAY_BAR_UNITS = {
        '1d': 1,
        '12h': 2,
        '6h': 4,
        '1h': 24,
        '30m': 48,
        '15m': 96
    }

    def __init__(
            self,
            market: Market,
            ticker: str,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime,
            normalizer: Normalizer
    ):
        assert set(intervals).issubset(set(self.supported_intervals)), 'Requested intervals are not supported.'

        super().__init__(market, ticker, intervals, features, start, end, normalizer)

    def __len__(self):
        return len(self.data[self.intervals[0]].index)

    def get_prices(self) -> np.array:
        return self.data['1d'].loc[:, 'Close']

    def get_item_shape(self) -> List[int]:
        interval_shape = 0
        for interval in self.current_intervals:
            interval_shape += self.INTERVAL_TO_DAY_BAR_UNITS[interval]

        features_shape = len(self.features)

        return [interval_shape, features_shape]

    @classmethod
    def get_day_bar_units_for(cls, interval: str) -> int:
        return cls.INTERVAL_TO_DAY_BAR_UNITS[interval]

    @property
    def current_intervals(self):
        return self.intervals

    @property
    def supported_intervals(self):
        return list(self.INTERVAL_TO_DAY_BAR_UNITS.keys())

    def __getitem__(self, day_index):
        item = np.empty(shape=(0, len(self.features)))
        for interval in self.current_intervals:
            bars_per_day = self.INTERVAL_TO_DAY_BAR_UNITS[interval]
            start_index = day_index * bars_per_day
            end_index = (day_index + 1) * bars_per_day

            features = self.data[interval][self.features].iloc[start_index: end_index].values
            features[..., self.num_price_features:] = self.normalizer(
                features[..., self.num_price_features:]
            )

            item = np.concatenate([
                item,
                features
            ])

        return item


class TrainValDayForecastDataset(TrainValMixin, DayForecastDataset):
    pass
