import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, List, Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from tables import NaturalNameWarning

from yacht import utils
from yacht.logger import Logger


class Market(ABC):
    """
        The role of the market is
            to provide data from some source,
            to cache that data for faster accessibility &
            to check the data correctness &
            to fill missing values
    """
    DOWNLOAD_MANDATORY_FEATURES = {
        'Close',
        'Open',
        'High',
        'Low',
        'Volume'
    }

    def __init__(
            self,
            get_features: List[str],
            logger: Logger,
            api_key,
            api_secret,
            storage_dir: str,
            include_weekends: bool,
            read_only: bool
    ):
        self.features = list(set(get_features).union(self.DOWNLOAD_MANDATORY_FEATURES))
        self.logger = logger
        self.api_key = api_key
        self.api_secret = api_secret
        self.include_weekends = include_weekends
        self.read_only = read_only

        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.mkdir(self.storage_dir)

        self.connection = self.open()

    @abstractmethod
    def open(self) -> Any:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def persist(self, interval: str):
        pass

    @abstractmethod
    def get(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime,
            features: Optional[List[str]] = None,
            squeeze: bool = False
    ) -> pd.DataFrame:
        """
            Returns: data within [start, end] and fills nan values.
            If squeeze = True, it should return [data.index.sort()[0] (=new_start), end],
                only if start <= new_start <= end, with nan values filled.
                The point of this is that some assets don't have data on the whole interval.
                This will move the starting point from where the data starts.
        """

        pass

    @abstractmethod
    def request(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime = None
    ) -> Union[List[List[Any]], pd.DataFrame]:
        """

        Args:
            ticker:
            interval:
            start:
            end:

        Returns:
            Prices data for [start, end] and interval: "interval"
        """

    @abstractmethod
    def process_request(self, data: Union[List[List[Any]], pd.DataFrame], **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def is_cached(self, ticker: str, interval: str, start: datetime, end: datetime) -> bool:
        pass

    @abstractmethod
    def cache_request(self, ticker: str, interval: str, data: pd.DataFrame):
        pass

    def download(
            self,
            tickers: Union[str, Iterable[str]],
            interval: str,
            start: datetime,
            end: datetime,
            squeeze: bool = False,
            **kwargs
    ):
        if isinstance(tickers, str):
            tickers = [tickers]

        warnings.filterwarnings(action='ignore', category=NaturalNameWarning)
        for ticker in tickers:
            self._download(ticker, interval, start, end, squeeze, **kwargs)
        warnings.filterwarnings(action='default', category=NaturalNameWarning)

    def _download(self, ticker: str, interval: str, start: datetime, end: datetime, squeeze: bool = False, **kwargs):
        # In some cases, we don't want to make rigid checks, only because there is no available data so far in the past.
        if squeeze:
            start, end = self.squeeze_period(ticker, interval, start, end)

        if self.is_cached(ticker, interval, start, end):
            return

        self.logger.info(f'[{interval}] - {ticker} - Downloading from "{start}" to "{end}"')

        data = self.request(ticker, interval, start, end)
        assert self.check_downloaded_data(data, interval, start, end), \
            f'[{ticker}] Download data did not passed the download checks.'
        data = self.process_request(data, **kwargs)
        data = data.sort_index()

        assert self.DOWNLOAD_MANDATORY_FEATURES.intersection(set(data.columns)) == self.DOWNLOAD_MANDATORY_FEATURES, \
            f'Some mandatory features are missing after downloading: {ticker}.'

        self.cache_request(ticker, interval, data)

    def squeeze_period(self, ticker: str, interval: str, start: datetime, end: datetime) -> Tuple[datetime, datetime]:
        """
            Try to squeeze the [start, end] period into [start', end'], where start <= start' & end' <= end.
            This is useful when we know that some data is not available.
        Args:
            ticker:
            interval:
            start:
            end:

        Returns:
            [start', end'], where start <= start' & end' <= end
        """
        return start, end
    
    def check_downloaded_data(
            self,
            data: Union[List[List[Any]], pd.DataFrame],
            interval: str,
            start: datetime,
            end: datetime
    ) -> bool:
        has_data = len(data) > 0

        return has_data


class H5Market(Market, ABC):
    def __init__(
            self,
            get_features: List[str],
            logger: Logger,
            api_key,
            api_secret,
            storage_dir: str,
            storage_file: str,
            include_weekends: bool,
            read_only: bool
    ):
        self.storage_file = os.path.join(storage_dir, storage_file)
        if read_only:
            assert os.path.exists(self.storage_file), 'In read only the h5 file should already exist.'

        # The is_cached operation is called multiple times. So we cache the data state for faster usage.
        # We cache in memory the disk cache state.
        self.is_cached_cache = dict()

        super().__init__(get_features, logger, api_key, api_secret, storage_dir, include_weekends, read_only)

    def open(self) -> pd.HDFStore:
        return pd.HDFStore(self.storage_file, mode='r' if self.read_only else 'a')

    def close(self):
        self.connection.close()

    def persist(self, interval: str):
        self.connection[interval].to_hdf(self.storage_file, interval, mode='w')

    @classmethod
    def create_key(cls, ticker: str, interval: str) -> str:
        return f'/{ticker}/{interval}'

    @classmethod
    def create_is_cached_key(cls, ticker: str, interval: str, start: datetime, end: datetime) -> str:
        return f'{ticker}@{interval}@{start}@{end}'

    def get(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime,
            features: Optional[List[str]] = None,
            squeeze: bool = False
    ) -> pd.DataFrame:
        """
            Returns: data within [start, end] and fills nan values.
            If squeeze = True, it should return [data.index.sort()[0] (=new_start), end],
                only if start <= new_start <= end, with nan values filled.
                The point of this is that some assets don't have data on the whole interval.
                This will move the starting point from where the data starts.
        """

        if squeeze:
            start, end = self.squeeze_period(ticker, interval, start, end)

        if not self.is_cached(ticker, interval, start, end):
            raise RuntimeError(f'[{ticker}]: "{interval}" not supported for {start} - {end}')

        data_slice = self._get(ticker, interval, start, end, features)
        data_slice.fillna(method='bfill', inplace=True, axis=0)
        data_slice.fillna(method='ffill', inplace=True, axis=0)

        assert np.isfinite(data_slice).all().all().item(), \
            f'Data from the market is not valid for: [{ticker}-{interval}] {start} - {end}'
        assert data_slice.index[0].date() == start.date() and data_slice.index[-1].date() == end.date(), \
            f'Data from the market is not valid for: [{ticker}-{interval}] {start} - {end}'

        return data_slice

    def _get(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime,
            features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Returns: data within [start, end].
        """

        if interval == '1h':
            # We need to add another day to get data for the interval [start, end].
            end = utils.add_days(end, action='+', include_weekends=self.include_weekends, offset=1)

        # Create the desired data span because there are missing values. In this way we will know exactly what data is
        # missing and at what index.
        datetime_index = utils.compute_period_range(start, end, self.include_weekends, interval)
        if features is None:
            features = self.features
        template_data = pd.DataFrame(index=datetime_index, columns=features)
        piece_of_data = self.connection[self.create_key(ticker, interval)].loc[start:end]
        template_data.update(piece_of_data)

        unsupported_features = set(features) - set(piece_of_data.columns)
        assert len(unsupported_features) == 0, \
            f'Unsupported features requested from the market: {unsupported_features}'

        return template_data

    def squeeze_period(self, ticker: str, interval: str, start: datetime, end: datetime) -> Tuple[datetime, datetime]:
        ticker_key = self.create_key(ticker, interval)
        if ticker_key in self.connection:
            indices = self.connection[ticker_key].loc[start:end].index
            if len(indices) == 0:
                return start, end

            new_start = indices[0]
            new_end = indices[-1]
            if new_start > start:
                start = new_start
                assert start <= end, \
                    f'Cannot move start after the end period: {start} > {end}'
            if new_end < end:
                end = new_end
                assert start <= end, \
                    f'Cannot move end before the start period: {start} > {end}'

        return start, end

    def is_cached(self, ticker: str, interval: str, start: datetime, end: datetime) -> bool:
        key = self.create_key(ticker, interval)
        if key not in self.connection:
            return False

        is_cached_key = self.create_is_cached_key(ticker, interval, start, end)
        if is_cached_key in self.is_cached_cache:
            return self.is_cached_cache[is_cached_key]

        if interval == '1h':
            # We need to add another day to get data for the interval [start, end].
            end = utils.add_days(end, action='+', include_weekends=self.include_weekends, offset=1)
        expected_period_range = utils.compute_period_range(start, end, self.include_weekends, interval)
        # Query how much of the requested time series is in the cache.
        actual_period_range = self.connection[key].loc[start:end].index

        # If we find almost all the asked dates we can say that the data is cached. We do not check for a
        # perfect match because the data from the API sometimes has leaks, therefore it would never be a match.
        # Also stocks have data only in the work days.
        is_cached_state = len(actual_period_range) >= 0.95 * len(expected_period_range)
        self.is_cached_cache[is_cached_key] = is_cached_state

        return is_cached_state

    def cache_request(self, ticker: str, interval: str, data: pd.DataFrame):
        key = self.create_key(ticker, interval)

        if key in self.connection:
            self.connection[key] = self.connection[key].combine_first(data)
        else:
            self.connection[key] = data
