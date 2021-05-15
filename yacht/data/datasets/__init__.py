from .base import *
from .day import *

from yacht import utils
from ..markets import build_market

dataset_registry = {
    'DayForecastDataset': DayForecastDataset
}


def build_dataset(input_config, storage_path) -> TradingDataset:
    market = build_market(input_config, storage_path)
    dataset_cls = dataset_registry[input_config.dataset]

    # TODO: Add multiple ticker support starting from here
    ticker = input_config.tickers[0]
    start = utils.string_to_datetime(input_config.start)
    end = utils.string_to_datetime(input_config.end)

    return dataset_cls(
        market=market,
        ticker=ticker,
        intervals=input_config.intervals,
        features=input_config.features,
        start=start,
        end=end
    )

