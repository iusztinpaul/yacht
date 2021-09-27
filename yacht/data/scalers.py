from abc import ABC, abstractmethod
from datetime import datetime
from typing import Union, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler as SkMinMaxScaler

from yacht.config import Config
from yacht.data.markets import Market


class Scaler(ABC):
    def __init__(self, ticker: str, features: List[str]):
        self.ticker = ticker
        self.features = features
        self.is_fitted = False

    def fit(self, data: Union[pd.DataFrame, np.ndarray]):
        if self.is_fitted is False:
            data = self._check_features(data)

            self._fit(data)

            # Avoid fitting multiple times to avoid useless computations.
            self.is_fitted = True

    @abstractmethod
    def _fit(self, data: Union[pd.DataFrame, np.ndarray]):
        pass

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        data = self._check_features(data)

        shape = None
        if isinstance(data, np.ndarray):
            shape = data.shape
            data = data.reshape(-1, shape[-1])

        data = self._transform(data)

        if shape is not None:
            data = data.reshape(shape)

        return data

    @abstractmethod
    def _transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass

    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        data = self._check_features(data)

        shape = None
        if isinstance(data, np.ndarray):
            shape = data.shape
            data = data.reshape(-1, shape[-1])

        data = self._inverse_transform(data)

        if shape is not None:
            data = data.reshape(shape)

        return data

    @abstractmethod
    def _inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        pass

    def _check_features(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            data = data[self.features]
        else:
            assert data.shape[-1] == len(self.features), 'Given data has more / less features than are supported.'

        return data

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
            end=train_end,
            flexible_start=True
        )
        scaler.fit(data)


class IdentityScaler(Scaler):
    def _fit(self, data: Union[pd.DataFrame, np.ndarray]):
        pass

    def _transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return data

    def _inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return data


class MinMaxScaler(Scaler):
    def __init__(self, ticker: str, features: List[str]):
        super().__init__(ticker=ticker, features=features)

        self.scaler = SkMinMaxScaler()

    def _fit(self, data: Union[pd.DataFrame, np.ndarray]):
        self.scaler.fit(data)

    def _transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        return self.scaler.transform(data)

    def _inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
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
        ticker=ticker,
        features=list(config.input.features) + list(config.input.technical_indicators),
    )
    scaler_singletones[ticker] = scaler

    return scaler
