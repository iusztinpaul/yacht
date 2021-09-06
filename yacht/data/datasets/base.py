from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import pandas as pd
from gym import Space, spaces
from pandas import Interval
from torch.utils.data import Dataset

from yacht import Mode
from yacht.data.markets import Market
from yacht.data.scalers import Scaler
from yacht.logger import Logger


class AssetDataset(Dataset, ABC):
    PRICE_FEATURES = (
        'Close',
        'Open',
        'High',
        'Low'
    )

    def __init__(
            self,
            market: Market,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            start: datetime,
            end: datetime,
            render_intervals: List[Interval],
            mode: Mode,
            logger: Logger,
            window_size: int = 1,
    ):
        """
            market:
            ticker:
            intervals: data bars frequency
            features: observation data features
            decision_price_feature: the feature that it will used for buying / selling assets or other decision making
            start:
            end:
            render_intervals: a list of datetime intervals to know if this environment should be rendered or not.
            normalizer:
            window_size: The past information that you want to add to the current item that you query from the dataset.
            data: If data != None, it will be encapsulated within the Dataset Object, otherwise it will be queried
                    from the market.
        """
        assert '1d' == intervals[0], 'One day bar interval is mandatory to exist & index=0 in input.intervals config.'
        assert 'Close' == features[0], 'Close feature/column is mandatory & index=0 in input.features config.'
        assert window_size >= 1

        self.market = market
        self.intervals = intervals
        self.features, self.price_features, self.other_features = self.split_features(features)
        self.decision_price_feature = decision_price_feature
        self.start = start
        self.end = end
        self.render_intervals = render_intervals
        self.mode = mode
        self.logger = logger
        self.window_size = window_size

        assert set(self.features) == set(self.price_features).union(set(self.other_features)), \
            '"self.features" should be all the supported features.'

    def close(self):
        self.market.close()

    @property
    def storage_dir(self) -> str:
        return self.market.storage_dir

    @property
    def include_weekends(self) -> bool:
        return self.market.include_weekends

    @property
    def num_price_features(self) -> int:
        return len(self.price_features)

    @property
    def num_other_features(self) -> int:
        return len(self.other_features)

    @property
    def should_render(self) -> bool:
        # Because it is not efficient to render all the environments, we choose over some desired logic what to render.
        for render_interval in self.render_intervals:
            if self.start in render_interval or self.end in render_interval:
                return True

        return False

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

    @property
    @abstractmethod
    def num_days(self) -> int:
        pass

    @property
    @abstractmethod
    def num_assets(self) -> int:
        pass

    @property
    @abstractmethod
    def asset_tickers(self) -> List[str]:
        pass

    @abstractmethod
    def index_to_datetime(self, integer_index: int) -> datetime:
        pass

    @abstractmethod
    def inverse_scaling(self, observation: dict, **kwargs) -> dict:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, current_index: int) -> Dict[str, np.array]:
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_prices(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_decision_prices(self, t_tick: Optional[int] = None, **kwargs) -> pd.Series:
        pass

    @abstractmethod
    def compute_mean_price(self, start: datetime, end: datetime) -> Union[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def get_external_observation_space(self) -> Dict[str, Space]:
        """
            Returns the gym spaces observation space in the format that the dataset gives the data.
        """
        pass


class SingleAssetDataset(AssetDataset, ABC):
    def __init__(
            self,
            ticker: str,
            market: Market,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            start: datetime,
            end: datetime,
            render_intervals: List[Interval],
            mode: Mode,
            logger: Logger,
            scaler: Scaler,
            window_size: int = 1,
            data: Dict[str, pd.DataFrame] = None
    ):
        super().__init__(
            market=market,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            start=start,
            end=end,
            render_intervals=render_intervals,
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        self.ticker = ticker
        self.scaler = scaler
        if data is not None:
            self.data = data
        else:
            self.logger.info(f'Preparing dataset... {ticker} | {start} to {end} | {intervals}')
            self.data = dict()
            for interval in self.intervals:
                self.market.download(ticker, interval, start, end)
                self.data[interval] = self.market.get(ticker, interval, start, end)
        self.prices = self.get_prices()

    def __str__(self) -> str:
        return self.ticker

    def __len__(self) -> int:
        return len(self.prices)

    @property
    def num_days(self) -> int:
        return len(self.data['1d'])

    @property
    def num_assets(self) -> int:
        return 1

    @property
    def asset_tickers(self) -> List[str]:
        return [self.ticker]

    def index_to_datetime(self, integer_index: int) -> datetime:
        return self.data['1d'].index[integer_index].to_pydatetime()

    def get_prices(self) -> pd.DataFrame:
        return self.data['1d']

    def get_decision_prices(self, t_tick: Optional[int] = None, **kwargs) -> pd.Series:
        if t_tick is None:
            decision_prices = self.prices.loc[slice(None), self.decision_price_feature]
            decision_prices.name = 'decision_price'
        else:
            t_datetime = self.index_to_datetime(t_tick)
            decision_prices = self.prices.loc[t_datetime, self.decision_price_feature]
            decision_prices = pd.Series(decision_prices, index=[self.ticker], name='decision_price')

        return decision_prices

    def compute_mean_price(self, start: datetime, end: datetime) -> Union[pd.DataFrame, pd.Series]:
        period_data = self.data['1d'].loc[start:end, self.decision_price_feature]
        period_mean = period_data.mean()

        return pd.Series(period_mean, index=[self.ticker], name='mean_price')

    def inverse_scaling(self, observation: dict, asset_idx: int = -1) -> dict:
        for interval in self.intervals:
            if asset_idx == -1:
                observation[interval] = self.scaler.inverse_transform(observation[interval])
            else:
                observation[interval][:, :, asset_idx, :] = self.scaler.inverse_transform(
                    observation[interval][:, :, asset_idx, :]
                )

        return observation


class MultiAssetDataset(AssetDataset):
    def __init__(
            self,
            datasets: List[SingleAssetDataset],
            market: Market,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            start: datetime,
            end: datetime,
            render_intervals: List[Interval],
            mode: Mode,
            logger: Logger,
            window_size: int = 1
    ):
        super().__init__(
            market=market,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            start=start,
            end=end,
            render_intervals=render_intervals,
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        self.datasets = datasets

        assert self.datasets[0].num_days * len(self.datasets) == sum([dataset.num_days for dataset in self.datasets]), \
            'All the datasets should have the same length.'

    @property
    def num_days(self) -> int:
        # All the datasets have the same number of days.
        return self.datasets[0].num_days

    @property
    def num_assets(self) -> int:
        return len(self.datasets)

    @property
    def asset_tickers(self) -> List[str]:
        return [dataset.ticker for dataset in self.datasets]

    def index_to_datetime(self, integer_index: int) -> datetime:
        # All the datasets have the same indices to dates mappings.
        return self.datasets[0].index_to_datetime(integer_index)

    def __len__(self):
        # All the datasets have the same length.
        return len(self.datasets[0])

    def __getitem__(self, current_index: int) -> Dict[str, np.array]:
        stacked_items: Dict[str, list] = defaultdict(list)
        for dataset in self.datasets:
            item = dataset[current_index]

            for key, value in item.items():
                stacked_items[key].append(value)

        for key, value in stacked_items.items():
            stacked_items[key] = np.stack(stacked_items[key], axis=2)

        return stacked_items

    def inverse_scaling(self, observation: dict, **kwargs) -> dict:
        for asset_idx in range(self.num_assets):
            dataset = self.datasets[asset_idx]
            observation = dataset.inverse_scaling(observation, asset_idx)

        return observation

    def __str__(self):
        asset_tickers = [ticker.split('-')[0] for ticker in self.asset_tickers]

        return '-'.join(asset_tickers)

    def get_prices(self) -> pd.DataFrame:
        prices = []
        for dataset in self.datasets:
            dataset_prices = dataset.get_prices()
            dataset_prices = dataset_prices.assign(ticker=dataset.ticker)
            dataset_prices = dataset_prices.set_index(keys=['ticker'], drop=True, append=True)

            prices.append(dataset_prices)

        prices = pd.concat(prices)

        return prices

    def get_decision_prices(self, t_tick: Optional[int] = None, ticker: Optional[str] = None) -> pd.Series:
        if ticker is not None:
            datasets = [self._pick_dataset(ticker=ticker)]
        else:
            datasets = self.datasets

        prices = []
        for dataset in datasets:
            decision_prices = dataset.get_decision_prices(t_tick)
            decision_prices.name = dataset.ticker

            prices.append(decision_prices)

        if t_tick is None:
            prices = pd.concat(prices, axis=1)  # We want to keep the dates as index.
        else:
            prices = pd.concat(prices, axis=0)  # We want to make the ticker as index.
        prices.name = 'decision_price'

        return prices

    def _pick_dataset(self, ticker: str) -> SingleAssetDataset:
        for dataset in self.datasets:
            if dataset.ticker == ticker:
                return dataset

        raise RuntimeError(f'No dataset with ticker: {ticker}')

    def compute_mean_price(self, start: datetime, end: datetime) -> Union[pd.DataFrame, pd.Series]:
        mean_data = []
        for dataset in self.datasets:
            mean_data.append(
                dataset.compute_mean_price(start, end).item()
            )

        return pd.Series(data=mean_data, index=[d.ticker for d in self.datasets])

    def get_external_observation_space(self) -> Dict[str, Space]:
        """
            Returns the gym spaces observation space in the format that the dataset gives the data.
        """
        observation_space = dict()
        # All the single asset observation spaces should have the same shapes.
        single_asset_observation_space = self.datasets[0].get_external_observation_space()
        for key, value in single_asset_observation_space.items():
            observation_space[key] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(*value.shape[:-1], len(self.datasets), value.shape[-1]),
                dtype=value.dtype
            )

        return observation_space


class IndexedDatasetMixin:
    def __init__(
            self,
            market: Market,
            ticker: str,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime,
            price_normalizer: Scaler,
            other_normalizer: Scaler,
            window_size: int,
            data: Dict[str, pd.DataFrame],
            indices: np.array
    ):
        """
            Mixin wrapper that is used with k_folding techniques to map indices within the data.
        """

        assert AssetDataset in type(self).mro(), \
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

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        index = self.getitem_index_mappings[index]

        return super().__getitem__(index)

    def get_prices(self) -> pd.DataFrame:
        prices = super().get_prices()
        prices = prices.iloc[self.getitem_index_mappings]

        return prices
