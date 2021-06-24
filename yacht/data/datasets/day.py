import logging
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from yacht.data.datasets import TradingDataset, IndexedDatasetMixin
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
            price_normalizer: Normalizer,
            other_normalizer: Normalizer,
            window_size: int = 1,
            data: Dict[str, pd.DataFrame] = None
    ):
        assert set(intervals).issubset(set(self.supported_intervals)), 'Requested intervals are not supported.'

        super().__init__(
            market,
            ticker,
            intervals,
            features,
            start,
            end,
            price_normalizer,
            other_normalizer,
            window_size,
            data
        )

    def __len__(self):
        return len(self.data[self.intervals[0]].index)

    def get_item_shape(self) -> List[int]:
        interval_shape = 0
        for interval in self.intervals:
            interval_shape += self.INTERVAL_TO_DAY_BAR_UNITS[interval]

        features_shape = len(self.features)

        return [interval_shape, features_shape]

    @classmethod
    def get_day_bar_units_for(cls, interval: str) -> int:
        return cls.INTERVAL_TO_DAY_BAR_UNITS[interval]

    @property
    def supported_intervals(self):
        return list(self.INTERVAL_TO_DAY_BAR_UNITS.keys())

    def __getitem__(self, day_index: int) -> Tuple[np.array, np.array]:
        window_item = []
        for i in reversed(range(self.window_size)):
            item = np.empty(shape=(0, len(self.features)))
            for interval in self.intervals:
                bars_per_day = self.INTERVAL_TO_DAY_BAR_UNITS[interval]
                start_index = (day_index - i) * bars_per_day
                end_index = (day_index - i + 1) * bars_per_day

                features = self.data[interval][self.features].iloc[start_index: end_index].values
                item = np.concatenate([
                    item,
                    features
                ], axis=0)

            window_item.append(item)

        window_item = np.stack(window_item, axis=0)
        unnormalized_window_item = window_item.copy()

        # Normalize data at window level.
        window_item[..., :self.num_price_features] = self.price_normalizer(
            window_item[..., :self.num_price_features]
        )
        window_item[..., self.num_price_features:] = self.other_normalizer(
            window_item[..., self.num_price_features:]
        )

        return window_item, unnormalized_window_item


class IndexedDayForecastDataset(IndexedDatasetMixin, DayForecastDataset):
    pass
