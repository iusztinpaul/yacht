import numpy as np

from typing import Tuple, Union

from data.loaders import BaseDataLoader
from environment.portfolios import Portfolio


class BaseEnvironment:
    def __init__(
            self,
            portfolio: Portfolio,
            reward_scheme,
            action_scheme,
            data_loader: BaseDataLoader
    ):
        self.portfolio = portfolio
        self.reward_scheme = reward_scheme
        self.action_scheme = action_scheme
        self.data_loader = data_loader

    @property
    def assets_num(self) -> int:
        return len(self.data_loader.tickers)

    @property
    def features_num(self) -> int:
        return len(self.data_loader.features)

    def next_batch(self) -> Union[np.array, Tuple[np.array]]:
        return self.data_loader.next_batch()

    def get_last_portfolio_weights(self):
        return self.portfolio.get_last_weights()

    def set_last_portfolio_weights(self, weights: np.array):
        return self.portfolio.set_last_weights(weights)
