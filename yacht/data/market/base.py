from datetime import datetime
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

    def get(self, start: datetime, end: datetime, ticker: str) -> np.array:
        """
            Makes a query for the interval [start, end) & ticker 'ticker',
            where the data frequency is the one from 'self.input_config.'
        Args:
            start: The start of the interval.
            end: The end of the interval.
            ticker: The ticker of the asset that you want to query

        Returns:
            A numpy array in the shape of features x time_span
        """
        raise NotImplementedError()

    def get_all(self, start: datetime, end: datetime) -> np.array:
        """
            Makes a query for the interval [start, end),
            where the data frequency is the one from 'self.input_config.'
        Args:
            start: The start of the interval.
            end: The end of the interval.

        Returns:
            A numpy array in the shape of features x assets x time_span
        """
        raise NotImplementedError()
