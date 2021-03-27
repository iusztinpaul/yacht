from datetime import datetime
from typing import Union, Tuple, List

import numpy as np

from config import Config
from data.loaders import build_data_loaders, BaseDataLoader
from data.market import build_market, BaseMarket
from environment.portfolios import build_portfolio, Portfolio


class Environment:
    def __init__(
            self,
            portfolio: Portfolio,
            reward_scheme,
            action_scheme,
            train_data_loader: BaseDataLoader,
            validation_data_loader: BaseDataLoader
    ):
        assert train_data_loader.market == validation_data_loader.market, \
            'Loaders should take data from the same market.'

        self.portfolio = portfolio
        self.reward_scheme = reward_scheme
        self.action_scheme = action_scheme
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader

    @property
    def market(self) -> BaseMarket:
        return self.train_data_loader.market

    def next_batch_train(self):
        return self._next_batch(self.train_data_loader)

    def next_batch_val(self):
        return self._next_batch(self.validation_data_loader)

    def _next_batch(self, data_loader: BaseDataLoader):
        X, y, batch_start_datetime = data_loader.next_batch()
        # TODO: Is it ok to sample data in this manner ?
        batch_previous_start_datetime = [
            last_start_datetime - data_loader.data_frequency_timedelta
            for last_start_datetime in batch_start_datetime
        ]
        batch_previous_w = self.portfolio.get_weights_at(batch_previous_start_datetime)

        return X, y, batch_previous_w, batch_start_datetime

    def set_portfolio_weights(
            self,
            index: Union[datetime, List[datetime]],
            weights: Union[np.array, List[np.array]]
    ):
        self.portfolio.set_weights_at(index, weights)


def build_environment(config: Config) -> Environment:
    market = build_market(config=config)
    training_data_loader, validation_data_loader = build_data_loaders(market=market, config=config)
    portfolio = build_portfolio(market=market, config=config)

    environment = Environment(
        portfolio=portfolio,
        reward_scheme=None,
        action_scheme=None,
        train_data_loader=training_data_loader,
        validation_data_loader=validation_data_loader
    )

    return environment
