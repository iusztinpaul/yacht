from abc import ABC
from datetime import datetime
from typing import List

import numpy as np
from torch.utils.data import Dataset

from yacht.data.markets import Market


class TradingDataset(Dataset, ABC):
    def __init__(
            self,
            market: Market,
            ticker: str,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime
    ):
        assert '1d' == intervals[0], 'One day bar interval is mandatory to exist & index=0 in input.intervals config.'
        assert 'Close' == features[0], 'Close feature/column is mandatory & index=0 in input.features config.'

        self.market = market
        self.ticker = ticker
        self.intervals = intervals
        self.features = features
        self.start = start
        self.end = end

    def close(self):
        self.market.close()

    def get_prices(self) -> np.array:
        raise NotImplementedError()

    def get_features_observation_space_shape(self) -> List[int]:
        raise NotImplementedError()
