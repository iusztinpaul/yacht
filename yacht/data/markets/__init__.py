import os

from .base import Market
from .binance import Binance, TechnicalIndicatorBinance
from .yahoo import Yahoo, TechnicalIndicatorYahoo

market_registry = {
    'Binance': Binance,
    'Yahoo': Yahoo,
    'TechnicalIndicatorYahoo': TechnicalIndicatorYahoo,
    'TechnicalIndicatorBinance': TechnicalIndicatorBinance,
}
singletones = dict()


def build_market(input_config, storage_path) -> Market:
    market_kwargs = {
        'features': list(input_config.features),
        'api_key': os.environ['MARKET_API_KEY'],
        'api_secret': os.environ['MARKET_API_SECRET'],
        'storage_dir': storage_path
    }
    market_name = input_config.market
    if len(input_config.technical_indicators) > 0:
        market_name = f'TechnicalIndicator{market_name}'
        market_kwargs['technical_indicators'] = list(input_config.technical_indicators)

    if market_name in singletones:
        return singletones[market_name]

    market_class = market_registry[market_name]
    market = market_class(**market_kwargs)
    singletones[input_config.market] = market

    return market
