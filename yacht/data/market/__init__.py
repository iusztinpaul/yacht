from .base import *
from .polonex import PoloniexMarket

from config import Config


markets = {
    'polonex': PoloniexMarket
}


def build_market(config: Config) -> BaseMarket:
    market_class = markets[config.input_config.market]
    market_object = market_class(config.input_config)

    return market_object
