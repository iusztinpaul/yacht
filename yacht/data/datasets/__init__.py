from typing import Union

from .base import *
from .aggregators import ChooseAssetDataset
from .multi_frequency import *

import yacht.utils as utils
from yacht.data.markets import build_market
from yacht.data.normalizers import build_normalizer
from yacht.config import Config
from ..renderers import TrainTestSplitRenderer

dataset_registry = {
    'DayMultiFrequencyDataset': DayMultiFrequencyDataset,
    'IndexedDayMultiFrequencyDataset': IndexedDayMultiFrequencyDataset
}


logger = logging.getLogger(__file__)


def build_dataset(config: Config, storage_dir, mode: str, render_split: bool = True) -> AssetDataset:
    assert mode in ('trainval', 'test')

    input_config = config.input
    assert len(input_config.tickers) > 0

    market = build_market(input_config, storage_dir)
    dataset_cls = dataset_registry[input_config.dataset]

    train_val_start, train_val_end, back_test_start, back_test_end = utils.split_period(
        input_config.start,
        input_config.end,
        input_config.backtest_split_ratio,
        input_config.backtest_embargo_ratio
    )

    # Download the whole requested interval in one shot for further processing & rendering.
    market.download(
        input_config.tickers,
        interval='1d',
        start=utils.string_to_datetime(input_config.start),
        end=utils.string_to_datetime(input_config.end)
    )
    if render_split:
        data = dict()
        for ticker in input_config.tickers:
            data[ticker] = market.get(
                ticker,
                '1d',
                utils.string_to_datetime(input_config.start),
                utils.string_to_datetime(input_config.end)
            )

        # Render de train-test split in rescaled mode.
        renderer = TrainTestSplitRenderer(
            data=data,
            train_interval=(train_val_start, train_val_end),
            test_interval=(back_test_start, back_test_end),
            rescale=True
        )
        renderer.render()
        renderer.save(utils.build_graphics_path(storage_dir, 'trainval_test_split_rescaled.png'))
        renderer.close()

        # Render de train-test split with original values.
        renderer = TrainTestSplitRenderer(
            data=data,
            train_interval=(train_val_start, train_val_end),
            test_interval=(back_test_start, back_test_end),
            rescale=False
        )
        renderer.render()
        renderer.save(utils.build_graphics_path(storage_dir, 'trainval_test_split.png'))
        renderer.close()

    logger.info(f'Trainval split: {train_val_start} - {train_val_end}')
    logger.info(f'Test split: {back_test_start} - {back_test_end}')

    if mode == 'trainval':
        start = train_val_start
        end = train_val_end
    else:
        start = back_test_start
        end = back_test_end

    datasets: List[SingleAssetTradingDataset] = []
    for ticker in input_config.tickers:
        price_normalizer = build_normalizer(input_config.price_normalizer)
        other_normalizer = build_normalizer(input_config.other_normalizer)

        datasets.append(
            dataset_cls(
                ticker=ticker,
                market=market,
                intervals=input_config.intervals,
                features=list(input_config.features) + list(input_config.technical_indicators),
                start=start,
                end=end,
                price_normalizer=price_normalizer,
                other_normalizer=other_normalizer,
                window_size=input_config.window_size
            )
        )

    return ChooseAssetDataset(
        datasets=datasets,
        market=market,
        intervals=input_config.intervals,
        features=list(input_config.features) + list(input_config.technical_indicators),
        start=start,
        end=end,
        window_size=input_config.window_size,
        default_ticker=input_config.tickers[0]
    )


def build_dataset_wrapper(dataset: AssetDataset, indices: List[int]) -> Union[IndexedDatasetMixin, AssetDataset]:
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
