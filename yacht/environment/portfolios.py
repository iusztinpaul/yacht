from typing import List

import numpy as np
import pandas as pd

from config import Config
from data.market import BaseMarket


class Portfolio:
    def __init__(self, tickers: List[str], time_span: int):
        """
        Args:
            tickers: List of the tickers that are relevant to the algorithm.
            time_span: Time period over which the weights will be persisted.
        """
        self.tickers = tickers
        self.time_span = time_span

        self.portfolio_vector_memory = pd.DataFrame(
            index=pd.RangeIndex(time_span),
            columns=tickers,
            dtype=np.float64
        )
        self.portfolio_vector_memory.fillna(1.0 / len(tickers), inplace=True)

    def get_last_weights(self):
        return self.get_weights_at(-1)

    def get_weights_at(self, index: int):
        return self.portfolio_vector_memory.iloc[index]

    def set_last_weights(self, weights: np.array):
        self.set_weights_at(-1, weights)

    def set_weights_at(self, index: int, weights: np.array):
        if weights.shape[0] != len(self.tickers):
            raise RuntimeError('Wrong number of weights distribution.')

        self.portfolio_vector_memory.iloc[index] = weights


def build_portfolio(market: BaseMarket, config: Config) -> Portfolio:
    portfolio = Portfolio(
        tickers=market.tickers,
        time_span=config.input_config.data_span
    )

    return portfolio
