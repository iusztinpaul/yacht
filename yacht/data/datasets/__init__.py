from .base import *
from .day import *

import yacht.utils as utils
from yacht.data.markets import build_market
from yacht.data.normalizers import build_normalizer
from yacht.data.k_fold import PurgedKFold

dataset_registry = {
    'DayForecastDataset': DayForecastDataset,
    'TrainValDayForecastDataset': TrainValDayForecastDataset
}


def build_train_val_dataset(input_config, storage_path, mode: str) -> TradingDataset:
    assert mode in ('train', 'val')

    market = build_market(input_config, storage_path)
    dataset_class_name = f'TrainVal{input_config.dataset}'
    dataset_cls = dataset_registry[dataset_class_name]
    normalizer = build_normalizer(input_config.dataset_normalizer)

    # TODO: Add multiple ticker support starting from here
    ticker = input_config.tickers[0]
    start = utils.string_to_datetime(input_config.start)
    end = utils.string_to_datetime(input_config.end)
    train_val_start, train_val_end, _, _ = \
        utils.split_period(start, end, input_config.back_test_split_ratio)

    k_fold = PurgedKFold(
            start=train_val_start,
            end=train_val_end,
            interval='1d',
            n_splits=3,
            embargo_ratio=0.01
        )

    return dataset_cls(
        market=market,
        ticker=ticker,
        intervals=input_config.intervals,
        features=input_config.features,
        start=train_val_start,
        end=train_val_end,
        normalizer=normalizer,
        k_fold=k_fold,
        mode=mode
    )


def build_back_test_dataset(input_config, storage_path) -> TradingDataset:
    market = build_market(input_config, storage_path)
    dataset_cls = dataset_registry[input_config.dataset]
    normalizer = build_normalizer(input_config.dataset_normalizer)

    # TODO: Add multiple ticker support starting from here
    ticker = input_config.tickers[0]
    start = utils.string_to_datetime(input_config.start)
    end = utils.string_to_datetime(input_config.end)
    _, _, back_test_start, back_test_end = \
        utils.split_period(start, end, input_config.back_test_split_ratio)

    return dataset_cls(
        market=market,
        ticker=ticker,
        intervals=input_config.intervals,
        features=input_config.features,
        start=back_test_start,
        end=back_test_end,
        normalizer=normalizer
    )
