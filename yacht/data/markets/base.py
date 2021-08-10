import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Union, List, Any

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
            include_weekends: bool
    ):
        self.features = list(set(features).union(self.MANDATORY_FEATURES))
        self.logger = logger
        self.api_key = api_key
        self.api_secret = api_secret
        self.include_weekends = include_weekends

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
    def get(self, ticker: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
            Returns: data within [start, end)
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

    def download(self, tickers: Union[str, List[str]], interval: str, start: datetime, end: datetime = None):
        if isinstance(tickers, str):
            tickers = [tickers]

        for ticker in tickers:
            self._download(ticker, interval, start, end)

    def _download(self, ticker: str, interval: str, start: datetime, end: datetime = None):
        if self.is_cached(ticker, interval, start, end):
            return

        self.logger.info(f'[{interval}] - {ticker} - Downloading from "{start}" to "{end}"')

        data = self.request(ticker, interval, start, end)
        data = self.process_request(data)

        assert self.MANDATORY_FEATURES.intersection(set(data.columns)) == self.MANDATORY_FEATURES, \
            'Some mandatory features are missing.'

        self.cache_request(ticker, interval, data)


class H5Market(Market, ABC):
    def __init__(
            self,
            features: List[str],
            logger: Logger,
            api_key,
            api_secret,
            storage_dir: str,
            storage_filename: str,
            include_weekends: bool
    ):
        self.storage_file = os.path.join(storage_dir, storage_filename)

        super().__init__(features, logger, api_key, api_secret, storage_dir, include_weekends)

    def open(self) -> pd.HDFStore:
        return pd.HDFStore(self.storage_file)

    def close(self):
        self.connection.close()

    def persist(self, interval: str):
        self.connection[interval].to_hdf(self.storage_file, interval, mode='w')

    @classmethod
    def create_key(cls, ticker: str, interval: str) -> str:
        return f'/{ticker}/{interval}'

    def get(self, ticker: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
            Returns: data within [start, end)
        """

        if not self.is_cached(ticker, interval, start, end):
            raise RuntimeError(f'Table: "{interval}" not supported')
        end = end - timedelta(microseconds=1)

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

        # Create the desired data span because there are missing values. In this way we will know exactly what data is
        # missing and at what index.
        date_time_index = pd.date_range(start=start, end=end, freq=freq)
        final_data = pd.DataFrame(index=date_time_index, columns=self.features)

        piece_of_data = self.connection[self.create_key(ticker, interval)].loc[start:end]
        final_data.update(piece_of_data)

        final_data.fillna(method='bfill', inplace=True, axis=0)
        final_data.fillna(method='ffill', inplace=True, axis=0)

        return final_data

    def is_cached(self, ticker: str, interval: str, start: datetime, end: datetime) -> bool:
        key = self.create_key(ticker, interval)
        if key not in self.connection:
            return False

        freq_mappings = {
            'd': 'D',
            'h': 'H',
            'm': 'min'
        }
        freq = f'{interval[:-1]}{freq_mappings[interval[-1]]}'
        time_series = pd.date_range(start, end, freq=freq)
        # Query how much of the requested time series is in the cache.
        valid_values = time_series[time_series.isin(self.connection[key].index)]

        # If we find almost all the asked dates we can say that the data is cached. We do not check for a
        # perfect match because the data from the API sometimes has leaks, therefore it would never be a match.
        # Also stocks have data only in the work days.
        return len(valid_values) >= 0.68 * len(time_series)

    def cache_request(self, ticker: str, interval: str, data: pd.DataFrame):
        key = self.create_key(ticker, interval)

        if key in self.connection:
            self.connection[key] = self.connection[key].combine_first(data)
        else:
            self.connection[key] = data
