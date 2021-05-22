import logging
from abc import ABC
from datetime import datetime
from typing import List, Tuple, Any, Dict

import numpy as np
from torch.utils.data import Dataset

from yacht.data.k_fold import PurgedKFold
from yacht.data.markets import Market
from yacht.data.normalizers import Normalizer


logger = logging.getLogger(__file__)


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

        assert set(self.features) == set(self.price_features).union(set(self.other_features)), \
            '"self.features" should be all the supported features.'

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

        return features, price_features, other_features,

    def get_prices(self) -> np.array:
        raise NotImplementedError()

    def get_item_shape(self) -> List[int]:
        raise NotImplementedError()


class TrainValMixin:
    def __init__(
            self,
            market: Market,
            ticker: str,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime,
            normalizer: Normalizer,
            k_fold: PurgedKFold,
            mode: str
    ):
        assert TradingDataset in type(self).mro(), \
            'IndexedMixin should be coupled with a TradingDataset class or subclass'
        assert mode in ('train', 'val')

        super().__init__(market, ticker, intervals, features, start, end, normalizer)

        self.k_fold = k_fold
        self.mode = mode

        if mode == 'train':
            self.getitem_index_mappings = next(k_fold.split(self.data['1d']))[0]
        else:
            self.getitem_index_mappings = next(k_fold.split(self.data['1d']))[1]

    def __getitem__(self, index: Dict[str, Any]):
        index = self.getitem_index_mappings[index]

        return super().__getitem__(index)

    def get_prices(self) -> np.array:
        prices = super().get_prices()
        prices = prices[self.getitem_index_mappings]

        return prices
