from typing import Union, Tuple

import numpy as np

from config import Config
from data.market import BaseMarket


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

        raise NotImplementedError()
