import logging
import os
from abc import ABC
from datetime import datetime
from typing import Union, List, Any

import pandas as pd

logger = logging.getLogger(__file__)


class Market(ABC):
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
            api_key,
            api_secret,
            storage_dir: str
    ):
        self.features = list(set(features).union(self.MANDATORY_FEATURES))
        self.api_key = api_key
        self.api_secret = api_secret

        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.mkdir(self.storage_dir)

        self.connection = self.open()

    def open(self) -> Any:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def persist(self, interval: str):
        raise NotImplementedError()

    def get(self, ticker: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
            Returns: data within [start, end)
        """

        raise NotImplementedError()

    def request(self, ticker: str, interval: str, start: datetime, end: datetime = None) -> List[List[Any]]:
        raise NotImplementedError()

    def process_request(self, data: List[List[Any]]) -> pd.DataFrame:
        raise NotImplementedError()

    def is_cached(self, interval: str, start: datetime, end: datetime) -> bool:
        raise NotImplementedError()

    def cache_request(self, interval: str, data: pd.DataFrame):
        raise NotImplementedError()

    def download(self, tickers: Union[str, List[str]], interval: str, start: datetime, end: datetime = None):
        if isinstance(tickers, str):
            tickers = [tickers]

        for ticker in tickers:
            self._download(ticker, interval, start, end)

    def _download(self, ticker: str, interval: str, start: datetime, end: datetime = None):
        if self.is_cached(interval, start, end):
            return

        logger.info(f'[{interval}] - {ticker} - Downloading from "{start}" to "{end}"')

        data = self.request(ticker, interval, start, end)
        data = self.process_request(data)

        self.cache_request(interval, data)
