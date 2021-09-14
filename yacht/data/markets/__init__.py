import os

from .base import Market
from .binance import Binance, TechnicalIndicatorBinance
from .yahoo import Yahoo, TechnicalIndicatorYahoo
from ...config import Config
from ...logger import Logger

market_registry = {
    'Binance': Binance,
    'Yahoo': Yahoo,
    'TechnicalIndicatorYahoo': TechnicalIndicatorYahoo,
    'TechnicalIndicatorBinance': TechnicalIndicatorBinance,
}


def build_market(config: Config, logger: Logger, storage_path: str) -> Market:
    input_config = config.input

    assert input_config.decision_price_feature, 'You have to pick a decision_price_feature.'

    market_kwargs = {
        'features': list(input_config.features) + [input_config.decision_price_feature],
        'logger': logger,
        'api_key': os.environ['MARKET_API_KEY'],
        'api_secret': os.environ['MARKET_API_SECRET'],
        'storage_dir': storage_path,
        'include_weekends': input_config.include_weekends
    }
    market_name = input_config.market
    if len(input_config.technical_indicators) > 0:
        market_name = f'TechnicalIndicator{market_name}'
        market_kwargs['technical_indicators'] = list(input_config.technical_indicators)
    market_kwargs['features'] = list(set(market_kwargs['features']))

    market_class = market_registry[market_name]
    market = market_class(**market_kwargs)

    return market
