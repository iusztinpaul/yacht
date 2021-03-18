from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

from config import Config
from data.market import BaseMarket


class Portfolio:
    def __init__(self, tickers: List[str], time_span: List[int]):
        """
        Args:
            tickers: List of the tickers that are relevant to the algorithm.
            time_span: Time interval over which the weights will be persisted.
        """
        self.tickers = ['Cash'] + tickers

        self.portfolio_vector_memory = pd.DataFrame(
            index=pd.to_datetime(time_span, unit='s'),
            columns=self.tickers,
            dtype=np.float64
        )
        self.portfolio_vector_memory.fillna(1.0 / len(tickers), inplace=True)

    def get_weights_at(self, index: Union[datetime, List[datetime]]) -> np.array:
        return np.array(self.portfolio_vector_memory.loc[index], dtype=np.float32)

    def set_weights_at(self, index: Union[datetime, List[datetime]], weights: np.array):
        if weights.shape[-1] != len(self.tickers):
            raise RuntimeError('Wrong number of weights distribution.')

        self.portfolio_vector_memory.loc[index] = weights


def build_portfolio(market: BaseMarket, config: Config) -> Portfolio:
    portfolio = Portfolio(
        tickers=market.tickers,
        time_span=config.input_config.data_span
    )

    return portfolio
