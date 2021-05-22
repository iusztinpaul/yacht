import os

from .base import Market
from .binance import Binance


market_registry = {
    'Binance': Binance
}
singletones = dict()


def build_market(input_config, storage_path) -> Market:
    if input_config.market in singletones:
        return singletones[input_config.market]

    market_class = market_registry[input_config.market]
    market = market_class(
        api_key=os.environ['MARKET_API_KEY'],
        api_secret=os.environ['MARKET_API_SECRET'],
        storage_dir=storage_path
    )
    singletones[input_config.market] = market

    return market
