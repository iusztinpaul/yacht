import logging
from datetime import datetime
from typing import List

import numpy as np

from yacht.data.datasets import TradingDataset
from yacht.data.markets import Market


logger = logging.getLogger(__file__)


class DayForecastDataset(TradingDataset):
    INTERVAL_TO_DAY_BAR_UNITS = {
        '1d': 1,
        '1h': 24,
        '30m': 48
    }
    NANOSECONDS_TO_BAR_UNIT_PER_DAY = {
        86400000000000: 1,  # 1 day.config.txt bar
        3600000000000: 24,  # 1 hour bar
        1800000000000: 2 * 24  # 30 min bar
    }

    def __init__(
            self,
            market: Market,
            ticker: str,
            intervals: List[str],
            start: datetime,
            end: datetime,
    ):
        assert '1d' in intervals, 'One day bar interval is mandatory.'
        assert set(intervals).issubset(set(self.supported_intervals)), 'Requested intervals are not supported.'

        super().__init__(market, ticker, intervals, start, end)

        logger.info(
            f'Downloading & loading data in memory for ticker - {ticker} - '
            f'from {start} to {end} - for intervals: {intervals}'
        )
        self.data = dict()
        for interval in intervals:
            self.market.download(ticker, interval, start, end)
            self.data[interval] = self.market.get(ticker, interval, start, end)

    def __len__(self):
        return len(self.data[self.intervals[0]].index)

    def get_prices(self) -> np.array:
        return self.data['1d'].loc[:, 'Close']

    def get_features_observation_space_shape(self) -> int:
        shape = 0
        for interval in self.current_intervals:
            shape += self.INTERVAL_TO_DAY_BAR_UNITS[interval]

        return shape

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
        features = np.empty(shape=(0, ))
        for interval in self.current_intervals:
            bars_per_day = self.INTERVAL_TO_DAY_BAR_UNITS[interval]
            start_index = day_index * bars_per_day
            end_index = (day_index + 1) * bars_per_day

            close_prices = self.data[interval].loc[:, 'Close'].values[start_index: end_index]

            features = np.concatenate([
                features,
                close_prices
            ])

        return features
