from .base import *
from .xy import XYDataLoader


def build_data_loader(market: BaseMarket, config: Config):
    data_loader = XYDataLoader(
        market=market,
        config=config
    )

    return data_loader
