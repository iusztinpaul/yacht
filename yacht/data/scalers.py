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
            features: List[str]
    ):
        if not scaler.is_fitted:
            data = market.get(
                ticker=scaler.ticker,
                interval=interval,
                start=train_start,
                end=train_end,
                features=features,
                squeeze=True
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


class TechnicalIndicatorMinMaxScaler(MinMaxScaler):
    def __init__(self, ticker: str, features: List[str]):
        super().__init__(ticker=ticker, features=features)

        self.scaler = SkMinMaxScaler()
        # TODO: Inject this list from the config for generalisation.
        self.supported_technical_indicators = ['rsi', 'macd', 'macds']
        self.features, self.identity_features = self._trim_features(self.features, self.supported_technical_indicators)

    @classmethod
    def _trim_features(cls, features: List[str], supported_features: List[str]):
        trimmed_features = set()
        for feature in features:
            # Make a more loose search in case that the names do not match perfectly.
            for supported_feature in supported_features:
                if supported_feature in feature.lower():
                    trimmed_features.add(feature)

        identity_features = set(features) - trimmed_features

        return list(trimmed_features), list(identity_features)

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        identity_data = data[self.identity_features]
        identity_data = identity_data.values
        transformed_data = super().transform(data)

        data = np.concatenate([identity_data, transformed_data], axis=-1)

        return data

    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        identity_data = data[self.identity_features]
        identity_data = identity_data.values
        transformed_data = super().inverse_transform(data)

        data = np.concat([identity_data, transformed_data], axis=-1)

        return data


#######################################################################################################################


scaler_registry = {
    'IdentityScaler': IdentityScaler,
    'MinMaxScaler': MinMaxScaler,
    'TechnicalIndicatorMinMaxScaler': TechnicalIndicatorMinMaxScaler
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
