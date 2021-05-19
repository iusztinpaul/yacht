from .base import *
from .day import *

from yacht import utils
from ..markets import build_market
from ..normalizers import build_normalizer

dataset_registry = {
    'DayForecastDataset': DayForecastDataset
}


def build_dataset(input_config, storage_path, train: bool) -> TradingDataset:
    market = build_market(input_config, storage_path)
    dataset_cls = dataset_registry[input_config.dataset]
    normalizer = build_normalizer(input_config.dataset_normalizer)

    # TODO: Add multiple ticker support starting from here
    ticker = input_config.tickers[0]
    start = utils.string_to_datetime(input_config.start)
    end = utils.string_to_datetime(input_config.end)
    train_start, train_end, back_test_start, back_test_end = \
        utils.split_period(start, end, input_config.back_test_split_ratio)
    if train:
        start = train_start
        end = train_end
    else:
        start = back_test_start
        end = back_test_end

    return dataset_cls(
        market=market,
        ticker=ticker,
        intervals=input_config.intervals,
        features=input_config.features,
        start=start,
        end=end,
        normalizer=normalizer
    )

