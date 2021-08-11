from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from yacht.config import Config
from yacht.data.markets import Market


class Scaler(ABC):
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.is_fitted = False

    def fit(self, data: Union[pd.DataFrame, np.ndarray]):
        if self.is_fitted is False:
            self._fit(data)

            # Avoid fitting multiple times to avoid useless computations.
            self.is_fitted = True

    @abstractmethod
    def _fit(self, data: Union[pd.DataFrame, np.ndarray]):
        pass

    @abstractmethod
    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass

    @abstractmethod
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass

    @classmethod
    def fit_on(
            cls,
            scaler: 'Scaler',
            market: Market,
            train_start: datetime,
            train_end: datetime,
            interval: str,
    ):
        data = market.get(
            ticker=scaler.ticker,
            interval=interval,
            start=train_start,
            end=train_end
        )
        scaler.fit(data)


class IdentityScaler(Scaler):
    def _fit(self, data: Union[pd.DataFrame, np.ndarray]):
        pass

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return data

    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return data


class MinMaxScaler(Scaler):
    def __init__(self, ticker: str):
        super().__init__(ticker=ticker)

        self.scaler = SkMinMaxScaler()

    def _fit(self, data: Union[pd.DataFrame, np.ndarray]):
        self.scaler.fit(data)

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return self.scaler.transform(data)

    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return self.scaler.inverse_transform(data)


#######################################################################################################################


scaler_registry = {
    'IdentityScaler': IdentityScaler,
    'MinMaxScaler': MinMaxScaler
}

scaler_singletones = dict()


def build_scaler(config: Config, ticker: str) -> Scaler:
    scaler_name = config.input.scaler

    if ticker in scaler_singletones:
        return scaler_singletones[ticker]

    scaler_class = scaler_registry[scaler_name]

    scaler = scaler_class(
        ticker=ticker
    )
    scaler_singletones[ticker] = scaler

    return scaler
