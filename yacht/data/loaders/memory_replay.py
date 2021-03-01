from datetime import datetime
from typing import Tuple, Union

import numpy as np

from config import Config
from data.loaders import BaseDataLoader
from data.market import BaseMarket
from data.memory_replay import MemoryReplayBuffer


class MemoryReplayDataLoader(BaseDataLoader):
    def __init__(
            self,
            market: BaseMarket,
            config: Config,
    ):
        super().__init__(market=market, config=config, window_size_offset=1)

        self.memory_replay = MemoryReplayBuffer.from_config(config)

    def next_batch(self) -> Union[np.array, Tuple[np.array]]:
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

        X = batch_data[:, :, :, :-1]
        y = batch_data[:, :, :, -1] / batch_data[:, 0, None, :, -2]

        return X, y

    def _end_datetime_from(self, start_datetime: datetime) -> datetime:
        window_size = self.config.input_config.window_size

        return start_datetime + (window_size + self.window_size_offset) * self.data_frequency_timedelta
