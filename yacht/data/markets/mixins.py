from typing import List, Any, Union, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from stockstats import StockDataFrame

from yacht import utils, errors
from yacht.config import Config


class TechnicalIndicatorMixin:
    def __init__(self, technical_indicators: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.technical_indicators = self.clean(technical_indicators)
        self.data_features = self.features
        self.features = self.technical_indicators + self.features

    def process_request(self, data: Union[List[List[Any]], pd.DataFrame], **kwargs) -> pd.DataFrame:
        df = super().process_request(data, **kwargs)

        stock = StockDataFrame.retype(df.copy())
        for technical_indicator_name in self.technical_indicators:
            oscillator_data = stock[technical_indicator_name]
            df[technical_indicator_name] = oscillator_data

        return df

    @staticmethod
    def clean(technical_indicators: List[str]) -> List[str]:
        def _clean(name: str) -> str:
            if 'LogDiff' in name:
                name = name.replace('LogDiff', '')
            if 'FracDiff' in name:
                name = name.replace('FracDiff', '')

            return name

        technical_indicators = {_clean(name) for name in technical_indicators}

        return list(technical_indicators)


class TargetPriceMixin:
    def process_request(self, data: Union[List[List[Any]], pd.DataFrame], **kwargs) -> pd.DataFrame:
        df = super().process_request(data, **kwargs)

        df['TP'] = df.apply(func=self.compute_target_price, axis=1)

        return df

    @classmethod
    def compute_target_price(cls, row: pd.Series):
        # Because there is no VWAP field in the yahoo data,
        # a method similar to Simpson integration is used to approximate VWAP.
        return (row['Open'] + 2 * row['High'] + 2 * row['Low'] + row['Close']) / 6


class LogDifferenceMixin:
    def process_request(self, data: Union[List[List[Any]], pd.DataFrame], **kwargs) -> pd.DataFrame:
        df = super().process_request(data, **kwargs)

        df_log_dif_t = df[self.DOWNLOAD_MANDATORY_FEATURES].copy()
        df_log_dif_t_minus_1 = df_log_dif_t.shift(1)
        # Add a small value to both sides to avoid division by 0 & log of 0.
        df_log_diff = (df_log_dif_t + 1e-7) / (df_log_dif_t_minus_1 + 1e-7)
        df_log_diff = df_log_diff.apply(np.log)
        df_log_diff.fillna(method='bfill', inplace=True, axis=0)
        df_log_diff.fillna(method='ffill', inplace=True, axis=0)

        log_diff_column_mappings = {column: f'{column}LogDiff' for column in self.DOWNLOAD_MANDATORY_FEATURES}
        df_log_diff.rename(columns=log_diff_column_mappings, inplace=True)

        df = pd.concat([df, df_log_diff], axis=1)

        return df


class FracDiffMixin:
    def process_request(self, data: Union[List[List[Any]], pd.DataFrame], config: Config, **kwargs) -> pd.DataFrame:
        df = super().process_request(data, **kwargs)

        # FIXME: Find a better place to process this stuff,
        #  because in this way we make the cached date config dependent.
        #  For now it is optimal to do it here because this is a time consuming operation & we don't want to do it
        #  at run time. When we change the code remove the 'config' parameter from the downloads methods.
        window_size = config.input.window_size
        train_split, _, _ = utils.split(
            config.input.start,
            config.input.end,
            config.input.validation_split_ratio,
            config.input.backtest_split_ratio,
            config.input.embargo_ratio,
            config.input.include_weekends
        )

        technical_indicators = TechnicalIndicatorMixin.clean(list(config.input.technical_indicators))
        features = self.DOWNLOAD_MANDATORY_FEATURES + technical_indicators
        data_to_process = df[features].copy()
        # Apply log on price features.
        data_to_process[self.DOWNLOAD_MANDATORY_FEATURES] += 1e-7  # To avoid log(0).
        data_to_process[self.DOWNLOAD_MANDATORY_FEATURES] = \
            data_to_process[self.DOWNLOAD_MANDATORY_FEATURES].apply(np.log)
        # Find the d_value only within the train_split.
        d_value = self.find_d_value(
            data=data_to_process.loc[train_split[0]: train_split[1]],
            size=window_size
        )

        assert d_value is not None, 'Could not find d_value'

        data_to_process = self.frac_diff_fixed_ffd(data=data_to_process, d=d_value, size=window_size)
        data_to_process.bfill(axis='rows', inplace=True)
        data_to_process.ffill(axis='rows', inplace=True)

        log_diff_column_mappings = {feature: f'{feature}FracDiff' for feature in features}
        data_to_process.rename(columns=log_diff_column_mappings, inplace=True)

        df = pd.concat([df, data_to_process], axis=1)

        return df

    @classmethod
    def find_d_value(cls, data: pd.DataFrame, size: int) -> Optional[float]:
        results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf'])
        for d in np.linspace(0, 2, 21):
            prices_df = data[['Close']].resample('1D').last()
            differentiated_df = cls.frac_diff_fixed_ffd(prices_df, d, size=size)
            try:
                differentiated_df = adfuller(differentiated_df['Close'], maxlag=1, regression='c', autolag=None)
            except ValueError:
                raise errors.PreProcessError()
            results.loc[d] = list(differentiated_df[:4]) + [differentiated_df[4]['5%']]
        
        return cls._parse_d_values(results)

    @classmethod
    def _parse_d_values(cls, results: pd.DataFrame) -> Optional[float]:
        # adfStat values are within [0, -inf]. When a adfStat value crosses the 95% conf border
        # we consider that we have found the d_value which makes the data stationary.
        conf_95 = results['95% conf'].mean()
        for d_value, row in results.iterrows():
            if row['adfStat'] <= conf_95:
                return d_value

        return None

    @classmethod
    def frac_diff_fixed_ffd(cls, data: pd.DataFrame, d: float, size: int) -> pd.DataFrame:
        # Constant width window
        w = cls.get_fixed_weights_ffd(d, size)
        width = len(w) - 1
        df = {}
        for name in data.columns:
            column_data = data[[name]].fillna(method='ffill').dropna()
            differentiated_column_data = pd.Series(dtype=np.float32)
            for end_iloc in range(width, column_data.shape[0]):
                start_loc = column_data.index[end_iloc - width]
                end_loc = column_data.index[end_iloc]
                if not np.isfinite(data.loc[end_loc, name]):
                    # Exclude NaNs
                    continue
                differentiated_column_data[end_loc] = np.dot(w.T, column_data.loc[start_loc:end_loc]).item()
            df[name] = differentiated_column_data.copy(deep=True)
        df = pd.concat(df, axis=1)

        return df

    @classmethod
    def get_fixed_weights_ffd(cls, d: float, size: int) -> np.ndarray:
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)

        return np.array(w[::-1], dtype=np.float32).reshape(-1, 1)
