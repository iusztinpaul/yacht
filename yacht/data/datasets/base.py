import logging
from abc import ABC
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from yacht.data.markets import Market
from yacht.data.normalizers import Normalizer


logger = logging.getLogger(__file__)


class TradingDataset(Dataset, ABC):
    PRICE_FEATURES = (
        'Close',
        'Open',
        'High',
        'Low'
    )

    def __init__(
            self,
            market: Market,
            ticker: str,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime,
            normalizer: Normalizer,
            data: Dict[str, pd.DataFrame] = None
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

        assert set(self.features) == set(self.price_features).union(set(self.other_features)), \
            '"self.features" should be all the supported features.'

        if data is not None:
            self.data = data
        else:
            logger.info(
                f'Downloading & loading data in memory for ticker - {ticker} - '
                f'from {start} to {end} - for intervals: {intervals}'
            )
            self.data = dict()
            for interval in intervals:
                self.market.download(ticker, interval, start, end)
                self.data[interval] = self.market.get(ticker, interval, start, end)

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
        for price_feature in cls.PRICE_FEATURES:
            if price_feature in features:
                price_features.append(price_feature)

        other_features = []
        for feature in features:
            if feature not in cls.PRICE_FEATURES:
                other_features.append(feature)

        # Always move price features at the beginning of the list, with `Close` price as the first one.
        features = price_features + other_features

        return features, price_features, other_features

    def get_folding_values(self) -> np.array:
        return self.get_prices()

    def get_prices(self) -> np.array:
        raise NotImplementedError()

    def get_item_shape(self) -> List[int]:
        raise NotImplementedError()


class IndexedDatasetMixin:
    def __init__(
            self,
            market: Market,
            ticker: str,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime,
            normalizer: Normalizer,
            data: Dict[str, pd.DataFrame],
            indices: List[int]
    ):
        assert TradingDataset in type(self).mro(), \
            'IndexedMixin should be coupled with a TradingDataset class or subclass'

        super().__init__(market, ticker, intervals, features, start, end, normalizer, data)

        self.getitem_index_mappings = indices

    def __len__(self):
        return len(self.getitem_index_mappings)

    def __getitem__(self, index: int):
        index = self.getitem_index_mappings[index]

        return super().__getitem__(index)

    def get_prices(self) -> np.array:
        prices = super().get_prices()
        prices = prices[self.getitem_index_mappings]

        return prices
