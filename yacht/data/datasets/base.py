from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from functools import cached_property
from typing import List, Dict, Union, Optional, Iterable

import numpy as np
import pandas as pd
from gym import Space, spaces
from pandas import Interval
from torch.utils.data import Dataset

from yacht import Mode, utils
from yacht.data.markets import Market
from yacht.data.scalers import Scaler
from yacht.data.transforms import Compose
from yacht.logger import Logger


class DatasetPeriod:
    def __init__(
            self,
            start: datetime,
            end: datetime,
            window_size: int,
            include_weekends: bool,
            take_action_at: str = 'current',
            frequency: str = 'd'
    ):
        assert frequency in ('d', )

        self.unadjusted_start = start
        self.unadjusted_end = end
        self.period_adjustment_size = self.compute_period_adjustment_size(
            window_size=window_size,
            take_action_at=take_action_at
        )
        # Adjust start with a 'window_size' length so we take data from the past & actually start from the given start.
        self.start = utils.adjust_period_with_window(
            datetime_point=start,
            window_size=self.period_adjustment_size,  # We also use the initial price within the period.
            action='-',
            include_weekends=include_weekends,
            frequency=frequency
        )
        self.end = end

        self.window_size = window_size
        self.include_weekends = include_weekends
        self.take_action_at = take_action_at
        self.frequency = frequency

        assert self.start <= self.unadjusted_start

    @classmethod
    def compute_period_adjustment_size(cls, window_size: int, take_action_at: str) -> int:
        assert take_action_at in ('current', 'next')

        if take_action_at == 'current':
            return window_size - 1
        elif take_action_at == 'next':
            return window_size

    def __len__(self) -> int:
        return utils.len_period_range(
            start=self.start,
            end=self.end,
            include_weekends=self.include_weekends
        )


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
            storage_dir: str,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            period: DatasetPeriod,
            render_intervals: List[Interval],
            mode: Mode,
            logger: Logger,
            window_size: int = 1,
    ):
        """
            market:
            storage_dir:
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
        assert window_size >= 1

        self.market = market
        self.storage_dir = storage_dir
        self.intervals = intervals
        self.features = features
        self.decision_price_feature = decision_price_feature
        self.render_intervals = render_intervals
        self.period = period
        self.mode = mode
        self.logger = logger
        self.window_size = window_size

    def close(self):
        self.market.close()

    @property
    def period_window_size(self) -> int:
        return self.period.window_size

    @property
    def period_adjustment_size(self) -> int:
        return self.period.period_adjustment_size

    @property
    def take_action_at(self) -> str:
        return self.period.take_action_at

    @property
    def first_observation_index(self) -> int:
        # Starting from 0 & the minimum value for the window_size is 1.
        return self.period_window_size - 1

    @property
    def last_observation_index(self) -> int:
        return self.period_adjustment_size + self.num_days - 1

    @property
    def unadjusted_start(self) -> datetime:
        return self.period.unadjusted_start

    @property
    def unadjusted_end(self) -> datetime:
        return self.period.unadjusted_end

    @property
    def start(self) -> datetime:
        return self.period.start

    @property
    def end(self) -> datetime:
        return self.period.end

    @property
    def include_weekends(self) -> bool:
        return self.market.include_weekends

    @cached_property
    def should_render(self) -> bool:
        # Because it is not efficient to render all the environments, we choose over some desired logic what to render.
        for render_interval in self.render_intervals:
            if self.start in render_interval or self.end in render_interval:
                return True

        return False

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
        """
        Args:
            current_index: The relative index the data will be given from.

        Returns:
            The data features within the [current_index - window_size + 1, current_index] interval.
        """
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
            storage_dir: str,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            period: DatasetPeriod,
            render_intervals: List[Interval],
            render_tickers: List[str],
            mode: Mode,
            logger: Logger,
            scaler: Scaler,
            window_transforms: Optional[Compose] = None,
            window_size: int = 1,
            data: Dict[str, pd.DataFrame] = None
    ):
        super().__init__(
            market=market,
            storage_dir=storage_dir,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            period=period,
            render_intervals=render_intervals,
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        self.ticker = ticker
        self.scaler = scaler
        self.window_transforms = window_transforms
        self.render_tickers = render_tickers

        if data is not None:
            self.data = data
        else:
            self.data = dict()
            for interval in self.intervals:
                self.data[interval] = self.market.get(
                    ticker=ticker,
                    interval=interval,
                    start=self.start,
                    end=self.end,
                    features=self.features + [self.decision_price_feature],
                    squeeze=False
                )
        self.prices = self.get_prices()

    def __str__(self) -> str:
        return self.ticker

    def __len__(self) -> int:
        # All the adjusted interval.
        return len(self.prices)

    @property
    def num_days(self) -> int:
        # Only the unadjusted interval.
        return utils.len_period_range(
            start=self.unadjusted_start,
            end=self.unadjusted_end,
            include_weekends=self.include_weekends
        )

    @property
    def num_assets(self) -> int:
        return 1

    @property
    def asset_tickers(self) -> List[str]:
        return [self.ticker]

    @cached_property
    def should_render(self) -> bool:
        if self.ticker in self.render_tickers:
            return super().should_render

        return False

    def index_to_datetime(self, integer_index: Union[int, Iterable]) -> Union[datetime, Iterable[datetime]]:
        return self.data['1d'].index[integer_index].to_pydatetime()

    def get_prices(self) -> pd.DataFrame:
        return self.market.get(
            ticker=self.ticker,
            interval='1d',
            start=self.start,
            end=self.end,
            features=list(self.market.DOWNLOAD_MANDATORY_FEATURES) + [self.decision_price_feature],
            squeeze=False
        )

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
    # TODO: Implement the multi-asset dependency within a DataFrame for faster processing.

    def __init__(
            self,
            datasets: List[SingleAssetDataset],
            storage_dir: str,
            market: Market,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            period: DatasetPeriod,
            render_intervals: List[Interval],
            render_tickers: List[str],
            mode: Mode,
            logger: Logger,
            window_size: int = 1,
            attached_datasets: Optional[List[SingleAssetDataset]] = None
    ):
        super().__init__(
            market=market,
            storage_dir=storage_dir,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            period=period,
            render_intervals=render_intervals,
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        self.datasets = datasets
        self.render_tickers = render_tickers
        self.attached_datasets = attached_datasets if attached_datasets is not None else []

        assert self.datasets[0].num_days * len(self.datasets) == sum([dataset.num_days for dataset in self.datasets]), \
            'All the datasets should have the same length.'

    @property
    def num_days(self) -> int:
        # All the datasets have the same number of days, because they are reflecting the same time (eg. the same month).
        return self.datasets[0].num_days

    @property
    def num_assets(self) -> int:
        return len(self.datasets)

    @property
    def asset_tickers(self) -> List[str]:
        return [dataset.ticker for dataset in self.datasets]

    @cached_property
    def should_render(self) -> bool:
        return any([dataset.should_render for dataset in self.datasets])

    def index_to_datetime(self, integer_index: Union[int, Iterable]) -> Union[datetime, Iterable[datetime]]:
        # All the datasets have the same indices to dates mappings.
        return self.datasets[0].index_to_datetime(integer_index)

    def __len__(self):
        # All the datasets have the same length.
        return len(self.datasets[0])

    def __getitem__(self, current_index: int) -> Dict[str, np.array]:
        datasets = self.datasets + self.attached_datasets
        stacked_items: Dict[str, list] = defaultdict(list)
        for dataset in datasets:
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

    def get_labels(self, t_tick: Optional[int] = None) -> Union[pd.DataFrame, pd.Series]:
        labels = []
        for dataset in self.datasets:
            ticker_labels = getattr(dataset, 'labels', pd.Series())
            ticker_labels.name = dataset.ticker
            labels.append(ticker_labels)
        labels = pd.concat(labels, axis=1)
        if len(labels) < t_tick:
            return pd.Series()

        if t_tick is not None:
            labels = labels.iloc[t_tick]

        return labels

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
                shape=(*value.shape[:-1], len(self.datasets + self.attached_datasets), value.shape[-1]),
                dtype=value.dtype
            )

        return observation_space
