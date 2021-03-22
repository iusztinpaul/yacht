import logging
import os
from datetime import datetime
from multiprocessing import Pool
from typing import Tuple

import numpy as np
from tqdm.contrib.concurrent import process_map

import utils
from config import InputConfig
from data.market import BaseMarket


logger = logging.getLogger(__file__)


class BaseDataLoader:
    def __init__(
            self,
            market: BaseMarket,
            input_config: InputConfig,
            window_size_offset: int = 1,
            max_workers: int = 4
    ):
        self.market = market
        self.input_config = input_config
        self.window_size_offset = window_size_offset
        self.max_workers = max_workers

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

    def next_batch(self) -> Tuple[np.array, np.array, list]:
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

        # batch_intervals = []
        # for _ in range(batch_size):
        #     # Add 'self.data_frequency_timedelta' because get_all() input is [start, end)
        #     batch_intervals.append(
        #         (start_datetime, end_datetime + self.data_frequency_timedelta)
        #     )
        #     start_datetime += self.data_frequency_timedelta
        #     end_datetime += self.data_frequency_timedelta
        #
        # logger.info(f'Loading a batch of {batch_size} from {batch_intervals[0][0]} to {batch_intervals[-1][1]}')
        # num_workers = batch_size if batch_size <= self.max_workers else self.max_workers
        # process_kwargs = {
        #     'max_workers': num_workers,
        #     'chunksize': utils.calc_chunksize(num_workers, batch_size)
        # }
        # batch_data = process_map(self._get_all_wrapper, batch_intervals, **process_kwargs)

        end_batch_datetime = start_datetime + (batch_size + self.input_config.window_size + self.window_size_offset - 1) * self.data_frequency_timedelta

        all_data = self.market.get_all(start_datetime, end_batch_datetime + self.data_frequency_timedelta)
        batch_data = []
        for i in range(batch_size):
            batch_data.append(all_data[..., i:i+self.input_config.window_size + 1])

        batch_start_datetimes = [start_datetime + self.data_frequency_timedelta * i for i in range(batch_size)]
        batch_data = np.stack(batch_data).astype(np.float32)

        X = batch_data[:, :, :, :-1]
        y = batch_data[:, :, :, -1] / batch_data[:, 0, None, :, -2]

        return X, y, batch_start_datetimes

    def compute_window_end_datetime(self, start_datetime: datetime) -> datetime:
        window_size = self.input_config.window_size

        return start_datetime + (window_size + self.window_size_offset - 1) * self.data_frequency_timedelta

    def _get_all_wrapper(self, params) -> np.array:
        start, end = params

        return self.market.get_all(start, end)
