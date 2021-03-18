from datetime import datetime
from typing import Union, Tuple

import numpy as np

from config import TrainingConfig, InputConfig
from data.market import BaseMarket


class BaseDataLoader:
    def __init__(
            self,
            market: BaseMarket,
            input_config: InputConfig,
            window_size_offset: int = 1
    ):
        self.market = market
        self.input_config = input_config
        self.window_size_offset = window_size_offset

    @property
    def data_frequency_timedelta(self):
        return self.input_config.data_frequency.timedelta

    @property
    def tickers(self):
        return self.market.tickers

    @property
    def features(self):
        return self.market.features

    def get_batch_size(self) -> int:
        raise NotImplementedError()

    def get_first_window_interval(self) -> Tuple[datetime, datetime]:
        raise NotImplementedError()

    def next_batch(self) -> Union[np.array, Tuple[np.array]]:
        """
            On every call it moves regarding to the rules of `self.memory_replay`
        Returns:
            A [batch_size, features, assets, (window_size + window_size_offset)] array
        """

        batch_size = self.get_batch_size()

        start_datetime, end_datetime = self.get_first_window_interval()

        if end_datetime > self.input_config.end_datetime:
            # TODO: Throw a custom Error.
            raise RuntimeError("Data stream finished.")

        batch_data = []
        batch_start_datetime = []
        for _ in range(batch_size):
            data_slice = self.market.get_all(start_datetime, end_datetime + self.data_frequency_timedelta)
            batch_data.append(data_slice)
            batch_start_datetime.append(start_datetime)

            start_datetime += self.data_frequency_timedelta
            end_datetime += self.data_frequency_timedelta
        batch_data = np.stack(batch_data).astype(np.float32)

        X = batch_data[:, :, :, :-1]
        y = batch_data[:, :, :, -1] / batch_data[:, 0, None, :, -2]

        return X, y, batch_start_datetime

    def compute_window_end_datetime(self, start_datetime: datetime) -> datetime:
        window_size = self.input_config.window_size

        return start_datetime + (window_size + self.window_size_offset - 1) * self.data_frequency_timedelta
