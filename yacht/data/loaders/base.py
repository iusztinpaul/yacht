from datetime import datetime
from typing import List

from configs.config import InputConfig


class BaseDataLoader:
    def __init__(
            self,
            tickers: List[str],
            input_config: InputConfig,
    ):
        self.tickers = tickers
        self.input_config = input_config
