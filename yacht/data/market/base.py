from datetime import datetime, timedelta
from typing import List

import numpy as np

from config.config import InputConfig


class BaseMarket:
    def __init__(self, input_config: InputConfig):
        self.input_config = input_config

    @property
    def tickers(self) -> List[str]:
        """
        Returns:
            A list of all supported tickers.
        """
        raise NotImplementedError()

    @property
    def features(self) -> List[str]:
        """
        Returns:
            A list of current used features.
        """
        raise NotImplementedError()

    @property
    def commission(self) -> float:
        return 0

    @property
    def max_download_timedelta(self) -> timedelta:
        return timedelta(weeks=5)

    def download(self, start: datetime, end: datetime):
        raise NotImplementedError()

    def get(self, start_dt: datetime, end_dt: datetime, ticker: str) -> np.array:
        """
            Makes a query for the interval [start, end] & ticker 'ticker',
            where the data frequency is the one from 'self.input_config.'
        Args:
            start_dt: The start of the interval.
            end_dt: The end of the interval.
            ticker: The ticker of the asset that you want to query

        Returns:
            A numpy array in the shape of features x time_span
        """
        raise NotImplementedError()

    def get_all(self, start_dt: datetime, end_dt: datetime) -> np.array:
        """
            Makes a query for the interval [start, end],
            where the data frequency is the one from 'self.input_config.'
        Args:
            start_dt: The start of the interval.
            end_dt: The end of the interval.

        Returns:
            A numpy array in the shape of features x assets x time_span
        """
        raise NotImplementedError()
