import os

from .base import Market
from .binance import Binance


market_registry = {
    'Binance': Binance
}


def build_market(input_config, storage_path):
    market_class = market_registry[input_config.market]

    return market_class(
        api_key=os.environ['MARKET_API_KEY'],
        api_secret=os.environ['MARKET_API_SECRET'],
        storage_dir=storage_path
    )
