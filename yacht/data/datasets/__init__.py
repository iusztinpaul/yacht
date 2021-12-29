import itertools
import time
from copy import copy
from typing import Set

from tqdm import tqdm

from .base import *
from .day_frequency import DayFrequencyDataset
from .teacher import TeacherDayFrequencyDataset
from .student import StudentMultiAssetDataset
from .samplers import SampleAssetDataset
from .multi_frequency import *

import yacht.utils as utils

from yacht.config import Config
from yacht.data import indexes
from yacht.data.markets import build_market
from yacht.data.renderers import TrainTestSplitRenderer
from yacht.data.scalers import build_scaler
from yacht.data.transforms import build_transforms
from yacht import Mode

dataset_registry = {
    'DayMultiFrequencyDataset': DayMultiFrequencyDataset,
    'DayFrequencyDataset': DayFrequencyDataset,
    'TeacherDayFrequencyDataset': TeacherDayFrequencyDataset
}


def build_dataset(
        config: Config,
        logger: Logger,
        storage_dir: str,
        mode: Mode,
        market_storage_dir: Optional[str] = None
) -> Optional[SampleAssetDataset]:
    start_building_data_time = time.time()

    input_config = config.input
    dataset_cls = dataset_registry[input_config.dataset]
    multi_asset_dataset_cls = StudentMultiAssetDataset if config.agent.is_student else MultiAssetDataset

    # -----------------------------------------------------------------------------------------------------------------
    # 1. Download
    # 2. Load
    # 3. Remove invalid tickers
    # -----------------------------------------------------------------------------------------------------------------
    is_read_only = market_storage_dir is not None
    tickers = build_tickers(config, mode)
    market = build_market(
        config=config,
        logger=logger,
        storage_dir=market_storage_dir if market_storage_dir is not None else storage_dir,
        read_only=is_read_only
    )
    start = utils.string_to_datetime(input_config.start)
    end = utils.string_to_datetime(input_config.end)
    if is_read_only is False:
        to_download_tickers = tickers
        # Indexes tickers are not used within the standard pipeline,
        # therefore we are adding them separately.
        to_download_tickers = to_download_tickers.union(set(input_config.indexes_tickers))
        # Download the whole requested interval in one shot for further processing & rendering.
        for interval in input_config.intervals:
            market.download(
                to_download_tickers,
                interval=interval,
                start=start,
                end=end,
                squeeze=True,
                config=config
            )
    # Remove tickers that are too sparse.
    num_tickers = len(tickers)
    tickers = remove_invalid_tickers(tickers, market, intervals=input_config.intervals, start=start, end=end)
    logger.info(f'Dropped {num_tickers - len(tickers)} corrupted tickers.')
    num_tickers = len(tickers)
    if num_tickers == 0:
        logger.error('No valid tickers to train on.')

        return None

    # -----------------------------------------------------------------------------------------------------------------
    # 1. Create splits.
    # 2. Render splits.
    # 3. Pick mode split.
    # -----------------------------------------------------------------------------------------------------------------
    train_split, validation_split, backtest_split = utils.split(
        input_config.start,
        input_config.end,
        input_config.validation_split_ratio,
        input_config.backtest_split_ratio,
        input_config.embargo_ratio,
        input_config.include_weekends
    )
    # Render split only for backtest tickers.
    if not mode.is_trainable() and config.meta.render_data_split:
        render_split(
            tickers=tickers,
            market=market,
            config=config,
            train_split=train_split,
            validation_split=validation_split,
            backtest_split=backtest_split,
            storage_dir=storage_dir,
            mode=mode
        )

    logger.info(f'Building datasets for: {mode.value}')
    logger.info(f'Loading the following assets [ num = {len(tickers)} ]:')
    logger.info(tickers)
    if mode.is_trainable() or mode.is_backtest_on_train():
        if config.agent.is_teacher:
            # In teacher mode, train over all the data. It is ok because it is used just to generate GT.
            train_split = utils.string_to_datetime(input_config.start), utils.string_to_datetime(input_config.end)

        start = train_split[0]
        end = train_split[1]
        logger.info(f'Train split: {start} - {end}')
    elif mode.is_validation():
        logger.info(f'Validation split: {validation_split[0]} - {validation_split[1]}')
        start = validation_split[0]
        end = validation_split[1]
    elif mode.is_test():
        logger.info(f'Backtest split: {backtest_split[0]} - {backtest_split[1]}')
        start = backtest_split[0]
        end = backtest_split[1]
    else:
        raise RuntimeError(f'Invalid mode for creating a split: {mode}')

    # -----------------------------------------------------------------------------------------------------------------
    # 1. Adjust split starting point.
    # 2. Compute dataset periods.
    # 3. Compute observation window_size.
    # -----------------------------------------------------------------------------------------------------------------
    assert bool(input_config.take_action_at) is True, '"input.take_action_at" feature is mandatory.'

    # Datasets will expand their data range with -window_size on the left side of the interval.
    start = utils.adjust_period_with_window(
        datetime_point=start,
        window_size=DatasetPeriod.compute_period_adjustment_size(
            window_size=input_config.window_size,
            take_action_at=input_config.take_action_at
        ),
        action='+',
        include_weekends=input_config.include_weekends
    )
    periods = utils.compute_periods(
        start=start,
        end=end,
        include_weekends=input_config.include_weekends,
        period_length=input_config.period_length,
        include_edges=False
    )

    total_num_datasets = len(periods) * len(list(itertools.combinations(tickers, config.input.num_assets_per_dataset)))
    logger.info('Creating datasets...')
    logger.info(f'Total estimated num datasets: {total_num_datasets}')
    if len(periods) == 0:
        logger.error(f'Num periods equal to 0. Dataset could not be created.')

        return None

    if config.agent.is_teacher:
        # In a teacher setup the agent will take the data from the whole period.
        # Make the window_size the maximum period length for observation size consistency.
        max_period_length = max([
            utils.len_period_range(
                start=period[0],
                end=period[1],
                include_weekends=input_config.include_weekends
            ) for period in periods
        ])
        window_size = input_config.window_size + max_period_length
    else:
        window_size = input_config.window_size

    # -----------------------------------------------------------------------------------------------------------------
    # 1. Build single asset datasets.
    # 2. Build asset scalers.
    # 3. Build multi asset datasets.
    # 4. Build dataset sampler.
    # -----------------------------------------------------------------------------------------------------------------
    render_intervals = utils.compute_render_periods(list(config.input.render_periods))
    render_tickers = list(input_config.render_tickers) if len(input_config.render_tickers) > 0 else list(tickers)[0]
    num_skipped_datasets = 0
    datasets: List[Union[SingleAssetDataset, MultiAssetDataset]] = []
    for (period_start, period_end) in tqdm(periods, desc='Num periods / Tickers'):
        dataset_period = DatasetPeriod(
            start=period_start,
            end=period_end,
            window_size=input_config.window_size,  # Same past offset for Student or Teacher setup.
            include_weekends=input_config.include_weekends,
            take_action_at=input_config.take_action_at,
            frequency='d'
        )
        for dataset_tickers in itertools.combinations(tickers, config.input.num_assets_per_dataset):
            # Also check data availability for every specific interval. The big interval might pass the tests, but
            # the local periods could not be valid.
            tickers_validity = [
                market.is_cached(ticker, interval, dataset_period.start, dataset_period.end)
                for ticker in dataset_tickers for interval in input_config.intervals
            ]
            if all(tickers_validity) is False:
                num_skipped_datasets += 1
                continue

            dataset_period = copy(dataset_period)
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
                    interval=config.input.scale_on_interval,
                    features=list(input_config.features) + list(input_config.technical_indicators)
                )

                transforms = build_transforms(config)
                single_asset_datasets.append(
                    dataset_cls(
                        ticker=ticker,
                        market=market,
                        storage_dir=storage_dir,
                        intervals=list(input_config.intervals),
                        features=list(input_config.features) + list(input_config.technical_indicators),
                        decision_price_feature=input_config.decision_price_feature,
                        period=dataset_period,
                        render_intervals=render_intervals,
                        render_tickers=render_tickers,
                        mode=mode,
                        logger=logger,
                        scaler=scaler,
                        window_transforms=transforms,
                        window_size=window_size
                    )
                )

            dataset = multi_asset_dataset_cls(
                datasets=single_asset_datasets,
                market=market,
                storage_dir=storage_dir,
                intervals=list(input_config.intervals),
                features=list(input_config.features) + list(input_config.technical_indicators),
                decision_price_feature=input_config.decision_price_feature,
                period=dataset_period,
                render_intervals=render_intervals,
                render_tickers=render_tickers,
                mode=mode,
                logger=logger,
                window_size=window_size
            )
            datasets.append(dataset)

    usable_num_datasets = total_num_datasets - num_skipped_datasets
    if usable_num_datasets == 0:
        return None

    period_length = len(utils.compute_period_range(
        start=periods[0][0],
        end=periods[0][1],
        include_weekends=input_config.include_weekends
    ))
    logger.info(f'Skipped {num_skipped_datasets} / {total_num_datasets} datasets.')
    logger.info(f'A total of {usable_num_datasets} datasets were created.')
    logger.info(f'Which is equal to a total of {usable_num_datasets * period_length} timesteps.')
    logger.info(f'Datasets built in {time.time() - start_building_data_time:.2f} seconds.')

    sample_dataset_period = DatasetPeriod(
        start=start,
        end=end,
        window_size=input_config.window_size,
        include_weekends=input_config.include_weekends,
        take_action_at=input_config.take_action_at,
        frequency='d'
    )
    return SampleAssetDataset(
        datasets=datasets,
        market=market,
        storage_dir=storage_dir,
        intervals=list(input_config.intervals),
        features=list(input_config.features) + list(input_config.technical_indicators),
        decision_price_feature=input_config.decision_price_feature,
        period=sample_dataset_period,
        render_intervals=render_intervals,
        mode=mode,
        logger=logger,
        window_size=window_size,
        default_index=0,
        shuffle=mode.is_trainable()
    )


def build_tickers(config: Config, mode: Mode) -> Set[str]:
    input_config = config.input

    if mode.is_download():
        tickers = set(input_config.fine_tune_tickers).union(set(input_config.tickers))
        tickers = tickers.union(set(input_config.backtest.tickers))
        tickers = list(tickers)
    elif mode.is_trainable():
        if mode.is_fine_tuning():
            tickers = list(input_config.fine_tune_tickers)
        else:
            tickers = list(input_config.tickers)
    else:
        tickers = list(input_config.backtest.tickers)

    assert len(tickers) > 0
    assert len(tickers) >= config.input.num_assets_per_dataset, 'Cannot create a dataset with less tickers than asked.'

    if 'S&P500' in tickers:
        tickers.remove('S&P500')
        tickers.extend(indexes.SP_500_TICKERS)
    if 'NASDAQ100' in tickers:
        tickers.remove('NASDAQ100')
        tickers.extend(indexes.NASDAQ_100_TICKERS)
    if 'DOW30' in tickers:
        tickers.remove('DOW30')
        tickers.extend(indexes.DOW_30_TICKERS)
    if 'RUSSELL2000' in tickers:
        tickers.remove('RUSSELL2000')
        tickers.extend(indexes.RUSSELL_2000_TICKERS)

    return set(tickers)


def remove_invalid_tickers(
        tickers: set,
        market: Market,
        intervals: List[str],
        start: datetime,
        end: datetime
) -> set:
    valid_tickers = set()
    for ticker in tickers:
        is_valid = []
        for interval in intervals:
            is_valid.append(
                market.is_cached(ticker, interval, start, end)
            )
        if all(is_valid):
            valid_tickers.add(ticker)

    return valid_tickers


def render_split(
        tickers: set,
        market: Market,
        config: Config,
        train_split: tuple,
        validation_split: tuple,
        backtest_split: tuple,
        storage_dir: str,
        mode: Mode
):
    input_config = config.input

    data = dict()
    for ticker in tickers:
        data[ticker] = market.get(
            ticker,
            '1d',
            utils.string_to_datetime(input_config.start),
            utils.string_to_datetime(input_config.end),
            squeeze=True
        )

    # Render de train-test split with their original values.
    renderer = TrainTestSplitRenderer(
        data=data,
        train_split=train_split,
        validation_split=validation_split,
        backtest_split=backtest_split,
        rescale=False
    )
    renderer.render()
    renderer.save(utils.build_graphics_path(storage_dir, f'{mode.value}_train_test_split.png'))
    renderer.close()

    # If there are more tickers also render the train-test split in rescaled mode.
    if len(data) > 1:
        renderer = TrainTestSplitRenderer(
            data=data,
            train_split=train_split,
            validation_split=validation_split,
            backtest_split=backtest_split,
            rescale=True
        )
        renderer.render()
        renderer.save(utils.build_graphics_path(storage_dir, f'{mode.value}_train_test_split_rescaled.png'))
        renderer.close()
