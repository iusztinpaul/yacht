import os

from .base import Market
from yacht.data.markets.binance import Binance
from yacht.data.markets.yahoo import Yahoo
from yacht.data.markets import mixins
from ...config import Config
from ...logger import Logger

market_registry = {
    'Binance': Binance,
    'Yahoo': Yahoo
}
mixins_registry = {
    'LogDifferenceMixin': mixins.LogDifferenceMixin,
    'FracDiffMixin': mixins.FracDiffMixin,
    'TargetPriceMixin': mixins.TargetPriceMixin
}


def build_market(config: Config, logger: Logger, storage_dir: str, read_only: bool) -> Market:
    input_config = config.input

    assert input_config.decision_price_feature, 'You have to pick a decision_price_feature.'

    market_kwargs = {
        'get_features': list(input_config.features) + [input_config.decision_price_feature],
        'logger': logger,
        'api_key': os.environ['MARKET_API_KEY'],
        'api_secret': os.environ['MARKET_API_SECRET'],
        'storage_dir': storage_dir,
        'include_weekends': input_config.include_weekends,
        'read_only': read_only
    }
    market_class = market_registry[input_config.market]
    market_mixins = [mixins_registry[name] for name in input_config.market_mixins]
    if len(input_config.technical_indicators) > 0:
        market_mixins.append(mixins.TechnicalIndicatorMixin)
        market_kwargs['technical_indicators'] = list(input_config.technical_indicators)
    market_class = type('EnhancedMarket', tuple(market_mixins) + (market_class, ), {})

    # Remove possible duplicates features.
    market_kwargs['get_features'] = list(set(market_kwargs['get_features']))

    market = market_class(**market_kwargs)

    return market
