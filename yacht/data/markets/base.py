import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, List, Any, Iterable

import pandas as pd

from yacht.logger import Logger


class Market(ABC):
    """
        The role of the market is
            to provide data from some source,
            to cache that data for faster accessibility &
            to check the data correctness &
            to fill missing values
    """
    MANDATORY_FEATURES = {
        'Close',
        'Open',
        'High',
        'Low',
        'Volume'
    }

    def __init__(
            self,
            features: List[str],
            logger: Logger,
            api_key,
            api_secret,
            storage_dir: str,
            include_weekends: bool,
            read_only: bool
    ):
        self.features = list(set(features).union(self.MANDATORY_FEATURES))
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
            flexible_start: bool = False
    ) -> pd.DataFrame:
        """
            Returns: data within [start, end] and fills nan values.
            If flexible_start = True, it should return [data.index.sort()[0], end] with nan values filled.
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
        pass

    @abstractmethod
    def process_request(self, data: Union[List[List[Any]], pd.DataFrame]) -> pd.DataFrame:
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
            flexible_start: bool = False
    ):
        if isinstance(tickers, str):
            tickers = [tickers]

        for ticker in tickers:
            self._download(ticker, interval, start, end, flexible_start)

    def _download(self, ticker: str, interval: str, start: datetime, end: datetime, flexible_start: bool = False):
        # In some cases, we don't want to make rigid checks, only because there is no available data so far in the past.
        if flexible_start:
            start = self.move_start(ticker, interval, start, end)

        if self.is_cached(ticker, interval, start, end):
            return

        self.logger.info(f'[{interval}] - {ticker} - Downloading from "{start}" to "{end}"')

        data = self.request(ticker, interval, start, end)
        assert self.check_downloaded_data(data, interval, start, end), \
            f'[{ticker}] Download data did not passed the download checks.'
        data = self.process_request(data)

        assert self.MANDATORY_FEATURES.intersection(set(data.columns)) == self.MANDATORY_FEATURES, \
            'Some mandatory features are missing.'

        self.cache_request(ticker, interval, data)

    @abstractmethod
    def move_start(self, ticker: str, interval: str, start: datetime, end: datetime) -> datetime:
        pass
    
    def check_downloaded_data(
            self,
            data: Union[List[List[Any]], pd.DataFrame],
            interval: str,
            start: datetime,
            end: datetime
    ) -> bool:
        return len(data) > 0

    def interval_to_pd_freq(self, interval: str) -> str:
        if self.include_weekends:
            database_to_pandas_freq = {
                'd': 'd',
                'h': 'h',
                'm': 'min'
            }
            freq = interval[:-1] + database_to_pandas_freq[interval[-1].lower()]
        else:
            # TODO: Adapt the business days logic to other intervals.
            assert interval == '1d'
            freq = 'B'

        return freq


class H5Market(Market, ABC):
    def __init__(
            self,
            features: List[str],
            logger: Logger,
            api_key,
            api_secret,
            storage_dir: str,
            storage_file: str,
            include_weekends: bool,
            read_only: bool
    ):
        self.storage_file = os.path.join(storage_dir, storage_file)
        # The is_cached operation is called multiple times. So we cache the data state for faster usage.
        # We cache in memory the disk cache state.
        self.is_cached_cache = dict()

        super().__init__(features, logger, api_key, api_secret, storage_dir, include_weekends, read_only)

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
    def create_is_cached_key(cls, ticker: str, start: datetime, end: datetime) -> str:
        return f'{ticker}@{start}@{end}'

    def get(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime,
            flexible_start: bool = False
    ) -> pd.DataFrame:
        """
            Returns: data within [start, end] and fills nan values.
            If flexible_start = True, it should return [data.index.sort()[0] (=new_start), end],
                only if new_start > given_start, with nan values filled.
        """

        # In some cases, we don't want to make rigid checks, only because there is no available data so far in the past.
        if flexible_start:
            start = self.move_start(ticker, interval, start, end)

        if not self.is_cached(ticker, interval, start, end):
            raise RuntimeError(f'[{ticker}]: "{interval}" not supported for {start} - {end}')

        data_slice = self._get(ticker, interval, start, end)
        data_slice.fillna(method='bfill', inplace=True, axis=0)
        data_slice.fillna(method='ffill', inplace=True, axis=0)

        assert data_slice.notna().all().all(), 'Data from the market is not valid.'

        return data_slice

    def _get(self, ticker: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
            Returns: data within [start, end].
        """

        # Create the desired data span because there are missing values. In this way we will know exactly what data is
        # missing and at what index.
        freq = self.interval_to_pd_freq(interval)
        date_time_index = pd.date_range(start=start, end=end, freq=freq)
        final_data = pd.DataFrame(index=date_time_index, columns=self.features)

        piece_of_data = self.connection[self.create_key(ticker, interval)].loc[start:end]
        final_data.update(piece_of_data)

        return final_data

    def move_start(self, ticker: str, interval: str, start: datetime, end: datetime) -> datetime:
        ticker_key = self.create_key(ticker, interval)
        if ticker_key in self.connection:
            new_start = self.connection[ticker_key].index[0]
            if new_start > start:
                start = new_start
                assert start < end, 'Cannot move start after the end period.'

        return start

    def is_cached(self, ticker: str, interval: str, start: datetime, end: datetime) -> bool:
        key = self.create_key(ticker, interval)
        if key not in self.connection:
            return False

        is_cached_key = self.create_is_cached_key(ticker, start, end)
        if is_cached_key in self.is_cached_cache:
            return self.is_cached_cache[is_cached_key]

        freq = self.interval_to_pd_freq(interval)
        time_series = pd.date_range(start, end, freq=freq)
        # Query how much of the requested time series is in the cache.
        valid_values = time_series[time_series.isin(self.connection[key].index)]

        # If we find almost all the asked dates we can say that the data is cached. We do not check for a
        # perfect match because the data from the API sometimes has leaks, therefore it would never be a match.
        # Also stocks have data only in the work days.
        is_cached_state = len(valid_values) >= 0.8 * len(time_series)
        self.is_cached_cache[is_cached_key] = is_cached_state

        return is_cached_state

    def cache_request(self, ticker: str, interval: str, data: pd.DataFrame):
        key = self.create_key(ticker, interval)

        if key in self.connection:
            self.connection[key] = self.connection[key].combine_first(data)
        else:
            self.connection[key] = data
