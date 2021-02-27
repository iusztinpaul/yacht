from .base import *
from config import Config
from data.loaders import build_data_loader
from data.market import build_market
from environment.portfolios import build_portfolio


def build_environment(config: Config) -> BaseEnvironment:
    market = build_market(config=config)
    data_loader = build_data_loader(market=market, config=config)
    portfolio = build_portfolio(market=market, config=config)

    # TODO: Change 'BaseEnvironment' to a more custom env.
    environment = BaseEnvironment(
        portfolio=portfolio,
        reward_scheme=None,
        action_scheme=None,
        data_loader=data_loader
    )

    return environment
