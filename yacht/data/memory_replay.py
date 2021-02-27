import numpy as np

from copy import copy
from datetime import datetime
from typing import List

from config import Frequency, Config


class MemoryReplayBuffer:
    def __init__(
            self,
            start: datetime,
            end: datetime,
            batch_size: int,
            windows_size: int,
            sample_bias: float,
            data_frequency: Frequency
    ):
        assert 0 < sample_bias <= 1

        self._start = start
        self._end = end
        self._batch_size = batch_size
        self._window_size = windows_size
        self._sample_bias = sample_bias
        self._data_frequency = data_frequency

        self._data_span = self._create_datetime_span()

    @staticmethod
    def from_config(config: Config):
        return MemoryReplayBuffer(
            start=config.input_config.start_datetime,
            end=config.input_config.end_datetime,
            batch_size=config.training_config.batch_size,
            windows_size=config.input_config.window_size,
            sample_bias=config.training_config.buffer_biased,
            data_frequency=config.input_config.data_frequency
        )

    def _create_datetime_span(self) -> List[datetime]:
        data_span = [copy(self._start)]

        current_date = self._start
        while current_date < self._end:
            current_date += self._data_frequency.timedelta
            data_span.append(current_date)

        return data_span

    def get_experience(self) -> datetime:
        random_index = self._sample_random_index(
            0,
            len(self._data_span) - max(self._batch_size, self._window_size)
        )

        return self._data_span[random_index]

    def _sample_random_index(self, start: int, end: int) -> int:
        random_index = np.random.geometric(self._sample_bias)
        while random_index > end - start:
            random_index = np.random.geometric(self._sample_bias)

        random_index = end - random_index

        return random_index
