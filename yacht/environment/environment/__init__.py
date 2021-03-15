from typing import Union, Tuple

import numpy as np

from config import Config
from data.loaders import build_data_loader, BaseDataLoader
from data.market import build_market, BaseMarket
from environment.portfolios import build_portfolio, Portfolio


class Environment:
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
    def market(self) -> BaseMarket:
        return self.data_loader.market

    @property
    def assets_num(self) -> int:
        return len(self.data_loader.tickers)

    @property
    def features_num(self) -> int:
        return len(self.data_loader.features)

    def next_batch(self) -> Union[np.array, Tuple[np.array]]:
        X, y = self.data_loader.next_batch()
        last_w = self.portfolio.get_last_weights()

        return X, y, last_w

    def get_last_portfolio_weights(self):
        return self.portfolio.get_last_weights()

    def set_last_portfolio_weights(self, weights: np.array):
        return self.portfolio.set_last_weights(weights)


def build_environment(config: Config) -> Environment:
    market = build_market(config=config)
    data_loader = build_data_loader(market=market, config=config)
    portfolio = build_portfolio(market=market, config=config)

    environment = Environment(
        portfolio=portfolio,
        reward_scheme=None,
        action_scheme=None,
        data_loader=data_loader
    )

    return environment
