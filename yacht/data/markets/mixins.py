from datetime import datetime
from typing import List, Any

import numpy as np
import pandas as pd
from stockstats import StockDataFrame

from yacht.data.markets import Market


class TechnicalIndicatorMixin:
    def __init__(self, technical_indicators: List[str], *args, **kwargs):
        assert Market in type(self).mro(), \
            '"TechnicalIndicatorMixin" works only with "Market" objects.'

        super().__init__(*args, **kwargs)

        self.technical_indicators = technical_indicators
        self.data_features = self.features
        self.features = self.technical_indicators + self.features

    def is_cached(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime
    ) -> bool:
        value = super().is_cached(ticker, interval, start, end)

        if value is False:
            return False

        key = self.create_key(ticker, interval)
        data_columns = set(self.connection[key].columns)

        return set(self.technical_indicators).issubset(data_columns) and value

    def process_request(self, data: List[List[Any]]) -> pd.DataFrame:
        df = super().process_request(data)

        stock = StockDataFrame.retype(df.copy())
        for technical_indicator_name in self.technical_indicators:
            oscillator_data = stock[technical_indicator_name]
            df[technical_indicator_name] = oscillator_data

        return df


class TargetPriceMixin:
    def process_request(self, data: List[List[Any]]) -> pd.DataFrame:
        df = super().process_request(data)

        df['TP'] = df.apply(func=self.compute_target_price, axis=1)

        return df

    @classmethod
    def compute_target_price(cls, row: pd.Series):
        # Because there is no VWAP field in the yahoo data,
        # a method similar to Simpson integration is used to approximate VWAP.
        return (row['Open'] + 2 * row['High'] + 2 * row['Low'] + row['Close']) / 6


class LogDifferenceMixin:
    def process_request(self, data: List[List[Any]]) -> pd.DataFrame:
        df = super().process_request(data)

        for column in self.DOWNLOAD_MANDATORY_FEATURES:
            df[f'{column}Diff'] = self.compute_log_diff(df[column])

        return df

    @classmethod
    def compute_log_diff(cls, column: pd.Series):
        column.fillna(method='bfill', inplace=True, axis=0)
        column.fillna(method='ffill', inplace=True, axis=0)

        data = column.values
        data = np.log(data)
        # Append the first item to keep consistency in data length.
        data = np.diff(data, prepend=data[0])

        return data
