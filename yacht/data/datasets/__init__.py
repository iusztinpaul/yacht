from .base import *
from .day import *

import yacht.utils as utils
from yacht.data.markets import build_market
from yacht.data.normalizers import build_normalizer

dataset_registry = {
    'DayForecastDataset': DayForecastDataset,
    'IndexedDayForecastDataset': IndexedDayForecastDataset
}


def build_dataset(input_config, train_config, storage_path, mode: str) -> TradingDataset:
    assert mode in ('trainval', 'test')

    market = build_market(input_config, storage_path)
    dataset_cls = dataset_registry[input_config.dataset]
    normalizer = build_normalizer(input_config.dataset_normalizer)

    # TODO: Add multiple ticker support starting from here
    ticker = input_config.tickers[0]
    train_val_start, train_val_end, back_test_start, back_test_end = utils.split_period(
        input_config.start,
        input_config.end,
        input_config.back_test_split_ratio,
        train_config.k_fold_embargo_ratio
    )

    if mode == 'trainval':
        start = train_val_start
        end = train_val_end
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
        normalizer=normalizer,
    )


def build_dataset_wrapper(dataset: TradingDataset, indices: List[int]) -> IndexedDatasetMixin:
    dataset_class_name = dataset.__class__.__name__
    dataset_class_name = f'Indexed{dataset_class_name}'
    dataset_class = dataset_registry[dataset_class_name]

    return dataset_class(
        dataset.market,
        dataset.ticker,
        dataset.intervals,
        dataset.features,
        dataset.start,
        dataset.end,
        dataset.normalizer,
        dataset.data,
        indices
    )
