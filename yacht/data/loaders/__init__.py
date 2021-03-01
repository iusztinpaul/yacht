from .base import *
from .memory_replay import MemoryReplayDataLoader


def build_data_loader(market: BaseMarket, config: Config):
    data_loader = MemoryReplayDataLoader(
        market=market,
        config=config
    )

    return data_loader
