import os.path
from typing import Union

from .base import *
from .day import *

import yacht.utils as utils
from yacht.data.markets import build_market
from yacht.data.normalizers import build_normalizer
from yacht.config import Config
from ..renderers import TrainTestSplitRenderer

dataset_registry = {
    'DayForecastDataset': DayForecastDataset,
    'IndexedDayForecastDataset': IndexedDayForecastDataset
}


def build_dataset(config: Config, storage_path, mode: str, render_split: bool = True) -> TradingDataset:
    assert mode in ('trainval', 'test')

    input_config = config.input

    market = build_market(input_config, storage_path)
    dataset_cls = dataset_registry[input_config.dataset]
    price_normalizer = build_normalizer(input_config.price_normalizer)
    other_normalizer = build_normalizer(input_config.other_normalizer)

    # TODO: Add multiple ticker support starting from here
    ticker = input_config.tickers[0]
    train_val_start, train_val_end, back_test_start, back_test_end = utils.split_period(
        input_config.start,
        input_config.end,
        input_config.back_test_split_ratio,
        input_config.back_test_embargo_ratio
    )

    if render_split:
        daily_prices = market.get(
            ticker=ticker,
            interval='1d',
            start=min(train_val_start, back_test_start),
            end=max(train_val_end, back_test_end)
        )
        daily_prices = daily_prices['Close']

        renderer = TrainTestSplitRenderer(
            prices=daily_prices,
            train_interval=(train_val_start, train_val_end),
            test_interval=(back_test_start, back_test_end)
        )
        renderer.render()
        renderer.save(os.path.join(storage_path, 'trainval_test_split.png'))
        renderer.close()

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
        price_normalizer=price_normalizer,
        other_normalizer=other_normalizer,
        window_size=input_config.window_size
    )


def build_dataset_wrapper(dataset: TradingDataset, indices: List[int]) -> Union[IndexedDatasetMixin, TradingDataset]:
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
        dataset.price_normalizer,
        dataset.other_normalizer,
        dataset.window_size,
        dataset.data,
        indices
    )
