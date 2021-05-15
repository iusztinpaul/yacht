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
            start: datetime,
            end: datetime
    ):
        self.market = market
        self.ticker = ticker
        self.intervals = intervals
        self.start = start
        self.end = end

    def close(self):
        self.market.close()

    def get_prices(self) -> np.array:
        raise NotImplementedError()

    def get_features_observation_space_shape(self) -> int:
        raise NotImplementedError()
