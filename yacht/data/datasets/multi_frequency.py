import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
from gym import spaces

from yacht.data.datasets import IndexedDatasetMixin, SingleAssetTradingDataset
from yacht.data.markets import Market
from yacht.data.normalizers import Normalizer

logger = logging.getLogger(__file__)


class DayMultiFrequencyDataset(SingleAssetTradingDataset):
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
            ticker: str,
            market: Market,
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
            ticker,
            market,
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

    def get_external_observation_space(self) -> Dict[str, spaces.Space]:
        observation_space = dict()
        for interval in self.intervals:
            interval_bars = self.INTERVAL_TO_DAY_BAR_UNITS[interval]
            observation_space[interval] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, interval_bars, len(self.features)),
                dtype=np.float32
            )

        return observation_space

    @classmethod
    def get_day_bar_units_for(cls, interval: str) -> int:
        return cls.INTERVAL_TO_DAY_BAR_UNITS[interval]

    @property
    def supported_intervals(self):
        return list(self.INTERVAL_TO_DAY_BAR_UNITS.keys())

    def __getitem__(self, day_index: int) -> Dict[str, np.array]:
        window_item = defaultdict(list)
        for i in reversed(range(self.window_size)):
            for interval in self.intervals:
                bars_per_day = self.INTERVAL_TO_DAY_BAR_UNITS[interval]
                start_index = (day_index - i) * bars_per_day
                end_index = (day_index - i + 1) * bars_per_day

                features = self.data[interval][self.features].iloc[start_index: end_index].values
                # Normalize data at window level.
                features[..., :self.num_price_features] = self.price_normalizer(
                    features[..., :self.num_price_features]
                )
                features[..., self.num_price_features:] = self.other_normalizer(
                    features[..., self.num_price_features:]
                )

                window_item[interval].append(features)
        else:
            for interval in self.intervals:
                window_item[interval] = np.stack(window_item[interval])

        return window_item


class IndexedDayMultiFrequencyDataset(IndexedDatasetMixin, DayMultiFrequencyDataset):
    pass
