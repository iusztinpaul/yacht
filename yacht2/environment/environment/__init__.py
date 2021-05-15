from datetime import datetime
from typing import Union, Tuple, List, Dict

import numpy as np

from config import Config
from data.loaders import BaseDataLoader, get_data_loader_builder
from data.market import build_market, BaseMarket
from data.renderers import build_renderer
from environments.portfolios import build_portfolio, Portfolio


class Environment:
    SUPPORTED_LOADER_KEYS = ['train', 'validation', 'back_test']

    def __init__(
            self,
            portfolio: Portfolio,
            reward_scheme,
            action_scheme,
            loaders: Dict[str, BaseDataLoader]
    ):
        self.loaders, self.loaders_market = self._assert_loaders(loaders)

        self.portfolio = portfolio
        self.reward_scheme = reward_scheme
        self.action_scheme = action_scheme

    def _assert_loaders(self, loaders: Dict[str, BaseDataLoader]) -> Tuple[Dict[str, BaseDataLoader], BaseMarket]:
        loader_items = list(loaders.items())

        def _assert(idx: int):
            previous_loader_key = loader_items[idx - 1][0]
            assert previous_loader_key in self.SUPPORTED_LOADER_KEYS, \
                f'Loader key: {previous_loader_key} not in {self.SUPPORTED_LOADER_KEYS}'

            assert isinstance(loader_items[idx - 1][1], BaseDataLoader)

        for idx in range(1, len(loaders.values())):
            _assert(idx - 1)
            _assert(idx)

            assert loader_items[idx - 1][1].market == loader_items[idx][1].market, \
                'Loaders should take data from the same market.'

        market = loader_items[0][1].market

        return loaders, market

    @property
    def market(self) -> BaseMarket:
        return self.loaders_market

    def next(self, reason: str):
        return self._next_batch(self.loaders[reason])

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


def build_environment(config: Config, reason: str) -> Environment:
    market = build_market(config=config)
    renderer = build_renderer()
    data_loader_builder = get_data_loader_builder(reason)
    loaders = data_loader_builder(
        market=market,
        renderer=renderer,
        config=config
    )
    portfolio = build_portfolio(market=market, config=config)

    environment = Environment(
        portfolio=portfolio,
        reward_scheme=None,
        action_scheme=None,
        loaders=loaders
    )

    return environment
