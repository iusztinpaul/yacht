from datetime import datetime
from typing import Union, Tuple

import numpy as np

from config import Config
from data.market import BaseMarket
from data.memory_replay import MemoryReplayBuffer


class BaseDataLoader:
    def __init__(
            self,
            market: BaseMarket,
            config: Config,
            window_size_offset: int = 1
    ):
        self.market = market
        self.config = config
        self.window_size_offset = window_size_offset

        self.memory_replay = MemoryReplayBuffer.from_config(config)

    @property
    def data_frequency_timedelta(self):
        return self.config.input_config.data_frequency.timedelta

    @property
    def tickers(self):
        return self.market.tickers

    @property
    def features(self):
        return self.market.features

    def next_batch(self) -> Union[np.array, Tuple[np.array]]:
        """
            On every call it moves regarding to the rules of `self.memory_replay`
        Returns:
            A [batch_size, features, assets, (window_size + window_size_offset)] array
        """
        batch_size = self.config.training_config.batch_size

        start_datetime = self.memory_replay.get_experience()
        end_datetime = self._end_datetime_from(start_datetime)

        if end_datetime > self.config.input_config.end_datetime:
            # TODO: Throw a custom Error.
            raise RuntimeError("Data stream finished.")

        batch_data = []
        for _ in range(batch_size):
            data_slice = self.market.get_all(start_datetime, end_datetime)
            batch_data.append(data_slice)

            start_datetime += self.data_frequency_timedelta
            end_datetime = self._end_datetime_from(start_datetime)

        batch_data = np.stack(batch_data)

        return batch_data

    def _end_datetime_from(self, start_datetime: datetime) -> datetime:
        window_size = self.config.input_config.window_size

        return start_datetime + (window_size + self.window_size_offset) * self.data_frequency_timedelta
