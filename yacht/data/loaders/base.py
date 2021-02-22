from typing import List

from configs.config import InputConfig
from data.market import BaseMarket


class BaseDataLoader:
    def __init__(
            self,
            market: BaseMarket,
            tickers: List[str],
            input_config: InputConfig,
    ):
        self.market = market
        self.tickers = tickers
        self.input_config = input_config

    def next_batch(self):
        raise NotImplementedError()
