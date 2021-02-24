from datetime import datetime
import numpy as np

from configs.config import InputConfig


class BaseMarket:
    def __init__(self, input_config: InputConfig):
        self.input_config = input_config

    def get(self, start: datetime, end: datetime, ticker: str) -> np.array:
        pass
