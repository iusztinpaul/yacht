from abc import ABC
from datetime import datetime
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

from yacht.data.markets import Market
from yacht.data.normalizers import Normalizer


class TradingDataset(Dataset, ABC):
    PRICE_FEATURES = {
        'Close',
        'Open',
        'High',
        'Low'
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
        assert '1d' == intervals[0], 'One day bar interval is mandatory to exist & index=0 in input.intervals config.'
        assert 'Close' == features[0], 'Close feature/column is mandatory & index=0 in input.features config.'

        self.market = market
        self.ticker = ticker
        self.intervals = intervals
        self.features, self.price_features, self.other_features = self.split_features(features)
        self.start = start
        self.end = end
        self.normalizer = normalizer

    def close(self):
        self.market.close()

    @property
    def num_price_features(self) -> int:
        return len(self.price_features)

    @property
    def num_other_features(self) -> int:
        return len(self.other_features)

    @classmethod
    def split_features(cls, features: List[str]) -> Tuple[List[str], List[str], List[str]]:
        price_features = []
        other_features = []
        for feature in features:
            if feature in cls.PRICE_FEATURES:
                price_features.append(feature)
            else:
                other_features.append(feature)

        # Always moving price features at the beginning of the list, with `Close` price as the first one.
        features = price_features + other_features

        return features, price_features, other_features,

    def get_prices(self) -> np.array:
        raise NotImplementedError()

    def get_item_shape(self) -> List[int]:
        raise NotImplementedError()
