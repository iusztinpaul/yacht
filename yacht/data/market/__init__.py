from config import Config
from .base import *
from .polonex import PoloniexMarket


def build_market(config: Config) -> BaseMarket:
    market_class = {
        'polonex': PoloniexMarket
    }[config.input_config.market]
    market_object = market_class(config.input_config)

    return market_object
