import logging
import time

from datetime import datetime
from typing import Tuple

import numpy as np

from config import InputConfig
from data.market import BaseMarket
from data.renderers import BaseRenderer

logger = logging.getLogger(__file__)


class BaseDataLoader:
    def __init__(
            self,
            market: BaseMarket,
            renderer: BaseRenderer,
            input_config: InputConfig,
            window_size_offset: int = 1,
            render_prices: bool = True
    ):
        self.market = market
        self.renderer = renderer
        self.input_config = input_config
        self.window_size_offset = window_size_offset

        logger.info(f'Starting loading all the data for: {self.__class__.__name__}')
        start_time = time.time()
        self.market.load_all()
        finish_time = time.time()
        logger.info(f'Cached all data in: {round(finish_time - start_time, 2)} seconds')

        if render_prices:
            self.renderer.time_series(
                self.market.assets,
                self.market.features,
                self.market.assets.index.names[0]
            )
            self.renderer.show()

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

    def get_first_batch_start_datetime(self) -> datetime:
        raise NotImplementedError()

    def get_last_batch_start_datetime(self, first_batch_start_datetime: datetime) -> datetime:
        batch_size = self.get_batch_size()

        return first_batch_start_datetime + (batch_size - 1) * self.data_frequency_timedelta

    def get_last_batch_end_datetime(self, first_batch_start_datetime: datetime) -> datetime:
        last_batch_start_datetime = self.get_last_batch_start_datetime(first_batch_start_datetime)
        window_timedelta = (self.input_config.window_size + self.window_size_offset) * self.data_frequency_timedelta

        return last_batch_start_datetime + window_timedelta

    def next_batch(self) -> Tuple[np.array, np.array, list]:
        """
            On every call it moves regarding to the get_first_batch_start_datetime() value`
        Returns:
            A [batch_size, features, assets, (window_size + window_size_offset)] array
        """

        first_batch_start_datetime = self.get_first_batch_start_datetime()
        last_batch_end_datetime = self.get_last_batch_end_datetime(first_batch_start_datetime)
        if last_batch_end_datetime > self.input_config.end_datetime:
            # TODO: Throw a custom Error.
            raise RuntimeError("Data stream finished.")

        all_data = self.market.get_all(
            first_batch_start_datetime,
            last_batch_end_datetime
        )

        batch_size = self.get_batch_size()
        batch_data = []
        for i in range(batch_size):
            batch_data.append(all_data[..., i:i + self.input_config.window_size + 1])

        batch_start_datetimes = [
            first_batch_start_datetime + self.data_frequency_timedelta * i for i in range(batch_size)
        ]
        batch_data = np.stack(batch_data).astype(np.float32)

        X = batch_data[:, :, :, :-1]
        y = batch_data[:, :, :, -1] / batch_data[:, 0, None, :, -2]

        return X, y, batch_start_datetimes
