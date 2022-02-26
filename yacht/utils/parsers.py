import os
import re
from datetime import datetime, timedelta
from functools import reduce
from typing import Union, List, Tuple

import pandas as pd
from pandas import Interval
from pandas._libs.tslibs.offsets import BDay

from yacht.config.proto.period_pb2 import PeriodConfig


def file_path_to_name(file_path: str) -> str:
    if not file_path:
        return file_path

    return os.path.split(file_path)[1].split('.')[0]


def string_to_datetime(string: str) -> datetime:
    # day/month/year
    return datetime.strptime(string, "%d/%m/%Y")


def interval_to_timedelta(string: str) -> timedelta:
    mappings = {
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '6h': timedelta(hours=6),
        '12h': timedelta(hours=12),
        '1d': timedelta(days=1)
    }

    return mappings[string]


def get_num_days(start: Union[str, datetime], end: Union[str, datetime], include_weekends: bool) -> int:
    """
        Returns the number of days between the interval [start, end).
    """

    if isinstance(start, str):
        start = string_to_datetime(start)
    if isinstance(end, str):
        end = string_to_datetime(end)

    if include_weekends:
        days = pd.date_range(start=start, end=end, freq='1d')
    else:
        days = pd.date_range(start=start, end=end, freq='B')

    # Do not include the last day.
    return len(days) - 1


def split(
        start: Union[str, datetime],
        end: Union[str, datetime],
        validation_split_ratio: float,
        backtest_split_ratio: float,
        embargo_ratio: float,
        include_weekends: bool = False,
        is_backtest_first: bool = False
) -> tuple:
    assert 0 < validation_split_ratio < 1
    assert 0 <= backtest_split_ratio < 1
    assert 0 < embargo_ratio < 1
    assert validation_split_ratio + backtest_split_ratio + 2 * embargo_ratio < 1

    has_backtest_split = backtest_split_ratio > 1e-7

    if isinstance(start, str):
        start = string_to_datetime(start)
    if isinstance(end, str):
        end = string_to_datetime(end)

    start_timestamp = start.timestamp()
    end_timestamp = end.timestamp()

    total_interval_length = end_timestamp - start_timestamp
    validation_split_length = validation_split_ratio * total_interval_length
    backtest_interval_length = backtest_split_ratio * total_interval_length
    embargo_offset_length = embargo_ratio * total_interval_length

    # If there is no backtest split, we need only one embargo ratio between the train & validation splits.
    needed_embargo_ratios = 2 if has_backtest_split else 1
    train_interval_length = \
        total_interval_length * \
        (1 - validation_split_ratio - backtest_split_ratio - needed_embargo_ratios * embargo_ratio)

    # TODO: Refactor those two if statements.
    if is_backtest_first is True and has_backtest_split is True:
        # TODO: Add embargo x2 after the test split.
        # Test -> Purge/Embargo -> Train -> Purge/Embargo -> Validation
        start_backtest_timestamp = start_timestamp
        end_backtest_timestamp = start_backtest_timestamp + backtest_interval_length
        start_backtest = datetime.fromtimestamp(start_backtest_timestamp)
        end_backtest = datetime.fromtimestamp(end_backtest_timestamp)

        start_train_timestamp = end_backtest_timestamp + embargo_offset_length
        end_train_timestamp = start_train_timestamp + train_interval_length
        start_train = datetime.fromtimestamp(start_train_timestamp)
        end_train = datetime.fromtimestamp(end_train_timestamp)

        start_validation_timestamp = end_train_timestamp + embargo_offset_length
        end_validation_timestamp = start_validation_timestamp + validation_split_length
        start_validation = datetime.fromtimestamp(start_validation_timestamp)
        end_validation = datetime.fromtimestamp(end_validation_timestamp)

        assert start == start_backtest and end == end_validation

        end_backtest = end_backtest.replace(hour=0, minute=0, second=0, microsecond=0)
        start_train = start_train.replace(hour=0, minute=0, second=0, microsecond=0)
        end_train = end_train.replace(hour=0, minute=0, second=0, microsecond=0)
        start_validation = start_validation.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # Train -> Purge/Embargo -> Validation -> Purge/Embargo -> Test
        start_train_timestamp = start_timestamp
        end_train_timestamp = start_timestamp + train_interval_length
        start_train = datetime.fromtimestamp(start_train_timestamp)
        end_train = datetime.fromtimestamp(end_train_timestamp)

        start_validation_timestamp = end_train_timestamp + embargo_offset_length
        end_validation_timestamp = start_validation_timestamp + validation_split_length
        start_validation = datetime.fromtimestamp(start_validation_timestamp)
        end_validation = datetime.fromtimestamp(end_validation_timestamp)

        if has_backtest_split:
            start_backtest_timestamp = end_validation_timestamp + embargo_offset_length
            end_backtest_timestamp = start_backtest_timestamp + backtest_interval_length
            start_backtest = datetime.fromtimestamp(start_backtest_timestamp)
            end_backtest = datetime.fromtimestamp(end_backtest_timestamp)
        else:
            start_backtest = datetime.fromtimestamp(end_timestamp)
            end_backtest = datetime.fromtimestamp(end_timestamp)

        assert start == start_train and end == end_backtest

        end_train = end_train.replace(hour=0, minute=0, second=0, microsecond=0)
        start_validation = start_validation.replace(hour=0, minute=0, second=0, microsecond=0)
        end_validation = end_validation.replace(hour=0, minute=0, second=0, microsecond=0)
        start_backtest = start_backtest.replace(hour=0, minute=0, second=0, microsecond=0)

    if not include_weekends:
        start_train = add_business_days(start_train, action='+', offset=1)
        end_train = add_business_days(end_train, action='-', offset=1)
        start_validation = add_business_days(start_validation, action='+', offset=1)
        end_validation = add_business_days(end_validation, action='-', offset=1)
        if has_backtest_split:
            start_backtest = add_business_days(start_backtest, action='+', offset=1)
            end_backtest = add_business_days(end_backtest, action='-', offset=1)

    return (start_train, end_train), (start_validation, end_validation), (start_backtest, end_backtest)


def adjust_period_with_window(
        datetime_point: Union[str, datetime],
        window_size: int,
        action: str,
        include_weekends: bool,
        frequency: str = 'd'
):
    assert action in ('+', '-')
    assert frequency in ('d', )

    if isinstance(datetime_point, str):
        datetime_point = string_to_datetime(datetime_point)

    if include_weekends:
        if action == '+':
            datetime_point += timedelta(days=window_size)
        else:
            datetime_point -= timedelta(days=window_size)
    else:
        datetime_point = add_business_days(datetime_point, action=action, offset=window_size)

    return datetime_point


def compute_periods(
        start: Union[str, datetime],
        end: Union[str, datetime],
        include_weekends: bool,
        period_length: str,
        include_edges: bool = False
) -> List[Tuple[datetime, datetime]]:
    assert period_length in ('all', '1M')

    if isinstance(start, str):
        start = string_to_datetime(start)
    if isinstance(end, str):
        end = string_to_datetime(end)

    if period_length == 'all':
        return [(start, end)]

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    offset = timedelta(days=1) if include_weekends else BDay(1)
    freq = {
        True: {
            '1M': '1MS'
        },
        False: {
            '1M': '1BMS'
        }
    }[include_weekends][period_length]

    month_periods = list(pd.interval_range(start=start, end=end, freq=freq))
    if len(month_periods) == 0:
        return []

    if include_edges:
        if month_periods[0].left != start:
            month_periods.insert(
                0, pd.Interval(left=start, right=month_periods[0].left)
            )
        if month_periods[-1].right != end:
            month_periods.append(
                pd.Interval(left=month_periods[-1].right, right=end)
            )

    periods = [
        (
            period.left.to_pydatetime(),
            (period.right - offset).to_pydatetime()
        ) for period in month_periods
    ]

    return periods


def len_period_range(start, end, include_weekends) -> int:
    return len(compute_period_range(start, end, include_weekends))


def compute_period_range(start, end, include_weekends, interval: str = '1d') -> List[datetime]:
    freq = interval_to_pd_freq(interval=interval, include_weekends=include_weekends)
    period_range = list(pd.date_range(start, end, freq=freq))
    if interval == '1h' and include_weekends is False:
        # The hours should be within [9:00, 15:00] when getting data within the trading hours.
        period_range = [item for item in period_range if item.hour != 16]

    return period_range


def interval_to_pd_freq(interval: str, include_weekends: bool) -> str:
    if include_weekends:
        database_to_pandas_freq = {
            'd': 'd',
            'h': 'h',
            'm': 'min'
        }
        freq = interval[:-1] + database_to_pandas_freq[interval[-1].lower()]
    else:
        assert interval in ('1d', '1h')
        database_to_pandas_freq = {
            '1d': 'B',
            '1h': 'BH'
        }
        freq = database_to_pandas_freq[interval]

    return freq


def compute_render_periods(config_periods: List[PeriodConfig]) -> List[Interval]:
    periods = []
    for config_period in config_periods:
        start = pd.Timestamp(string_to_datetime(config_period.start))
        end = pd.Timestamp(string_to_datetime(config_period.end))

        periods.append(
            pd.Interval(left=start, right=end, closed='both')
        )

    return periods


def add_days(
        obj: Union[datetime, pd.Timestamp],
        action: str,
        include_weekends: bool,
        offset: int = 1
) -> datetime:
    assert action in ('+', '-')

    if include_weekends:
        if action == '+':
            return obj + timedelta(days=offset)
        else:
            return obj - timedelta(days=offset)
    else:
        return add_business_days(obj, action, offset)


def add_business_days(obj: Union[datetime, pd.Timestamp], action: str, offset: int = 1) -> datetime:
    assert action in ('+', '-')

    if isinstance(obj, datetime):
        obj = pd.Timestamp(obj)
    if action == '+':
        obj += BDay(offset)
    else:
        obj -= BDay(offset)

    return obj.to_pydatetime()


def english_title_to_snake_case(string: str) -> str:
    return reduce(lambda x, y: x + ('_' if y == ' ' else y), string).lower()


def camel_to_snake(string: str) -> str:
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)

    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()
