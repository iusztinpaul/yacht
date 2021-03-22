import numpy as np

from datetime import datetime
from typing import Tuple, Union

from config import InputConfig, TrainingConfig
from data.loaders import BaseDataLoader
from data.market import BaseMarket
from data.memory_replay import MemoryReplayBuffer


class TrainDataLoader(BaseDataLoader):
    def __init__(
            self,
            market: BaseMarket,
            input_config: InputConfig,
            training_config: TrainingConfig
    ):
        super().__init__(
            market=market,
            input_config=input_config,
            window_size_offset=1
        )

        self.training_config = training_config
        self.memory_replay = MemoryReplayBuffer.from_config(input_config, training_config)

    def get_batch_size(self) -> int:
        return self.training_config.batch_size

    def get_first_window_interval(self) -> Tuple[datetime, datetime]:
        start_datetime = self.memory_replay.get_experience()
        end_datetime = self.compute_window_end_datetime(start_datetime)

        return start_datetime, end_datetime


class ValidationDataLoader(BaseDataLoader):
    def __init__(
            self,
            market: BaseMarket,
            input_config: InputConfig,
    ):
        super().__init__(
            market=market,
            input_config=input_config,
            window_size_offset=1
        )

        self.X = None
        self.y = None
        self.batch_start_datetimes = None

    def get_batch_size(self) -> int:
        return len(self.input_config.data_span)

    def get_first_window_interval(self) -> Tuple[datetime, datetime]:
        start_datetime = self.input_config.start_datetime
        end_datetime = self.compute_window_end_datetime(start_datetime)

        return start_datetime, end_datetime

    def next_batch(self) -> Tuple[np.array, np.array, list]:
        if self.has_cached_data:
            return self.X, self.y, self.batch_start_datetimes

        X, y, batch_start_datetimes = super().next_batch()
        self.X = X
        self.y = y
        self.batch_start_datetimes = batch_start_datetimes

        return X, y, batch_start_datetimes

    @property
    def has_cached_data(self) -> bool:
        return self.X is not None and self.y is not None and self.batch_start_datetimes is not None
