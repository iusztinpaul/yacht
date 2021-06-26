from datetime import datetime
from typing import List, Any

import pandas as pd
from stockstats import StockDataFrame

from yacht.data.markets import Market


class TechnicalIndicatorMixin:
    def __init__(self, technical_indicators: List[str], *args, **kwargs):
        assert Market in type(self).mro(), \
            '"OscillatorMixin" works only with "Market" objects.'

        super().__init__(*args, **kwargs)

        self.technical_indicators = technical_indicators
        self.data_features = self.features
        self.features = self.technical_indicators + self.features

    def is_cached(self, interval: str, start: datetime, end: datetime) -> bool:
        value = super().is_cached(interval, start, end)

        if interval not in self.connection:
            return value

        data_columns = set(self.connection[interval].columns)

        return set(self.technical_indicators).issubset(data_columns) and value

    def process_request(self, data: List[List[Any]]) -> pd.DataFrame:
        df = super().process_request(data)

        stock = StockDataFrame.retype(df.copy())
        for technical_indicator_name in self.technical_indicators:
            oscillator_data = stock[technical_indicator_name]
            df[technical_indicator_name] = oscillator_data

        return df


