import itertools

from .base import *
from .aggregators import ChooseAssetDataset
from .multi_frequency import *

import yacht.utils as utils
from yacht.data.markets import build_market
from yacht.data.scalers import build_scaler
from yacht.config import Config
from ..renderers import TrainTestSplitRenderer
from ... import Mode

dataset_registry = {
    'DayMultiFrequencyDataset': DayMultiFrequencyDataset,
    'IndexedDayMultiFrequencyDataset': IndexedDayMultiFrequencyDataset
}
split_rendered = False


def build_dataset(
        config: Config,
        logger: Logger,
        storage_dir,
        mode: Mode,
        render_split: bool = True
) -> ChooseAssetDataset:
    global split_rendered

    input_config = config.input
    tickers = input_config.tickers if mode.is_trainable() else input_config.backtest.tickers
    assert len(tickers) > 0

    market = build_market(config, logger, storage_dir)
    dataset_cls = dataset_registry[input_config.dataset]

    train_split, validation_split, backtest_split = utils.split_period(
        input_config.start,
        input_config.end,
        input_config.validation_split_ratio,
        input_config.backtest_split_ratio,
        input_config.embargo_ratio
    )

    # Download the whole requested interval in one shot for further processing & rendering.
    market.download(
        tickers,
        interval='1d',
        start=utils.string_to_datetime(input_config.start),
        end=utils.string_to_datetime(input_config.end)
    )
    # Render the split only once. It is computationally useless to rendered it multiple times.
    if not split_rendered and render_split:
        split_rendered = True

        data = dict()
        for ticker in tickers:
            data[ticker] = market.get(
                ticker,
                '1d',
                utils.string_to_datetime(input_config.start),
                utils.string_to_datetime(input_config.end)
            )

        # Render de train-test split in rescaled mode.
        renderer = TrainTestSplitRenderer(
            data=data,
            train_split=train_split,
            validation_split=validation_split,
            backtest_split=backtest_split,
            rescale=True
        )
        renderer.render()
        renderer.save(utils.build_graphics_path(storage_dir, 'train_test_split_rescaled.png'))
        renderer.close()

        # Render de train-test split with original values.
        renderer = TrainTestSplitRenderer(
            data=data,
            train_split=train_split,
            validation_split=validation_split,
            backtest_split=backtest_split,
            rescale=False
        )
        renderer.render()
        renderer.save(utils.build_graphics_path(storage_dir, 'train_test_split.png'))
        renderer.close()

    logger.info(f'Train split: {train_split[0]} - {train_split[1]}')
    logger.info(f'Validation split: {validation_split[0]} - {validation_split[1]}')
    logger.info(f'Backtest split: {backtest_split[0]} - {backtest_split[1]}')

    if mode.is_trainable() or mode.is_backtest_on_train():
        start = train_split[0]
        end = train_split[1]
    elif mode.is_validation():
        start = validation_split[0]
        end = validation_split[1]
    else:
        start = backtest_split[0]
        end = backtest_split[1]

    datasets: List[Union[SingleAssetDataset, MultiAssetDataset]] = []
    for _ in itertools.combinations(tickers, config.input.num_assets_per_dataset):
        dataset_tickers = np.random.choice(
            tickers,
            config.input.num_assets_per_dataset,
            replace=False
        )
        single_asset_datasets: List[SingleAssetDataset] = []

        for ticker in dataset_tickers:
            scaler = build_scaler(
                config=config,
                ticker=ticker
            )
            Scaler.fit_on(
                scaler=scaler,
                market=market,
                train_start=train_split[0],
                train_end=train_split[1],
                interval=config.input.scale_on_interval
            )

            single_asset_datasets.append(
                dataset_cls(
                    ticker=ticker,
                    market=market,
                    intervals=input_config.intervals,
                    features=list(input_config.features) + list(input_config.technical_indicators),
                    start=start,
                    end=end,
                    mode=mode,
                    logger=logger,
                    scaler=scaler,
                    window_size=input_config.window_size
                )
            )

        dataset = MultiAssetDataset(
            datasets=single_asset_datasets,
            market=market,
            intervals=input_config.intervals,
            features=list(input_config.features) + list(input_config.technical_indicators),
            start=start,
            end=end,
            mode=mode,
            logger=logger,
            window_size=input_config.window_size
        )
        datasets.append(dataset)

    return ChooseAssetDataset(
        datasets=datasets,
        market=market,
        intervals=input_config.intervals,
        features=list(input_config.features) + list(input_config.technical_indicators),
        start=start,
        end=end,
        mode=mode,
        logger=logger,
        window_size=input_config.window_size,
        default_index=0
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
