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
            price_normalizer: Normalizer,
            other_normalizer: Normalizer,
            window_size: int = 1,
            data: Dict[str, pd.DataFrame] = None
    ):
        """
            market:
            ticker:
            intervals: data bars frequency
            features: supported data features
            start:
            end:
            normalizer:
            window_size: The past information that you want to add to the current item that you query from the dataset.
            data: If data != None, it will be encapsulated within the Dataset Object, otherwise it will be queried
                    from the market.
        """
        assert '1d' == intervals[0], 'One day bar interval is mandatory to exist & index=0 in input.intervals config.'
        assert 'Close' == features[0], 'Close feature/column is mandatory & index=0 in input.features config.'
        assert window_size >= 1

        self.market = market
        self.ticker = ticker
        self.intervals = intervals
        self.features, self.price_features, self.other_features = self.split_features(features)
        self.start = start
        self.end = end
        self.price_normalizer = price_normalizer
        self.other_normalizer = other_normalizer
        self.window_size = window_size

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
    def storage_dir(self) -> str:
        return self.market.storage_dir

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

    def __getitem__(self, current_index: int):
        raise NotImplementedError()

    def get_k_folding_values(self) -> np.array:
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
            price_normalizer: Normalizer,
            other_normalizer: Normalizer,
            window_size: int,
            data: Dict[str, pd.DataFrame],
            indices: np.array
    ):
        """
            Mixin wrapper that is used with k_folding techniques to map indices within the data.
        """

        assert TradingDataset in type(self).mro(), \
            'IndexedMixin should be coupled with a TradingDataset class or subclass'
        assert len(indices.shape) == 1

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

        self.getitem_index_mappings = self._adjust_indices_to_window(indices)

    def _adjust_indices_to_window(self, indices: np.array) -> np.array:
        """
            A window fashion aggregation of data does not allow discontinuity in the indices. For example if you have
            the following indices: [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15] with a window_size of 2 and you want to get
            an item start from index 5 the naive approach will take indices [9, 10], where 9 is not within this split.
            Because of that this adjusting finds the point of discontinuity and removes items from that point equal to
            the window size. Therefore, in our example the new list will be: [0, 1, 2, 3, 4, 11, 12, 13, 14, 15].
            Now the result for index 5 will be [10, 11].
        """
        delta = np.diff(indices)
        # Shift items one unit to the right for consistency between delta & indices.
        delta = np.concatenate([np.ones(shape=(1, )), delta], axis=0)
        discontinuity_starting_point = np.where(delta != 1)[0]

        assert len(discontinuity_starting_point) in (0, 1), 'No more than 1 discontinuity point is supported'
        if len(discontinuity_starting_point) == 0:
            return indices

        discontinuity_starting_point = discontinuity_starting_point[0]
        indices = np.concatenate([
            indices[:discontinuity_starting_point], indices[(discontinuity_starting_point + 1 + self.window_size):]
            ], axis=-1)

        return indices

    def __len__(self) -> int:
        return len(self.getitem_index_mappings)

    def __getitem__(self, index: int) -> np.array:
        index = self.getitem_index_mappings[index]

        return super().__getitem__(index)

    def get_prices(self) -> np.array:
        prices = super().get_prices()
        prices = prices[self.getitem_index_mappings]

        return prices
