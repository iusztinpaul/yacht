from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from gym import spaces
from pandas import Interval

from yacht import Mode
from yacht.data.datasets import SingleAssetDataset, DatasetPeriod
from yacht.data.markets import Market
from yacht.data.scalers import Scaler
from yacht.logger import Logger


class DayMultiFrequencyDataset(SingleAssetDataset):
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
            storage_dir: str,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            period: DatasetPeriod,
            render_intervals: List[Interval],
            mode: Mode,
            logger: Logger,
            scaler: Scaler,
            window_size: int = 1,
            data: Dict[str, pd.DataFrame] = None
    ):
        assert set(intervals).issubset(set(self.supported_intervals)), 'Requested intervals are not supported.'

        super().__init__(
            ticker=ticker,
            market=market,
            storage_dir=storage_dir,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            period=period,
            render_intervals=render_intervals,
            mode=mode,
            logger=logger,
            scaler=scaler,
            window_size=window_size,
            data=data
        )

    def __len__(self):
        return len(self.data['1d'])

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
        """
        Args:
            day_index: The relative index the data will be given from.

        Returns:
            The data features within the [day_index - window_size + 1, day_index] interval.
        """

        window_item: Dict[str, Union[list, np.ndarray]] = defaultdict(list)
        for i in reversed(range(self.window_size)):
            for interval in self.intervals:
                bars_per_day = self.INTERVAL_TO_DAY_BAR_UNITS[interval]
                start_index = (day_index - i) * bars_per_day
                end_index = (day_index - i + 1) * bars_per_day
                features = self.data[interval][self.features].iloc[start_index: end_index].values

                window_item[interval].append(features)
        else:
            for interval in self.intervals:
                window_item[interval] = np.stack(window_item[interval])
                window_item[interval] = self.scaler.transform(window_item[interval])

        return window_item
