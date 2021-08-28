import os
from datetime import datetime, timedelta
from functools import reduce
from typing import Union

import pandas as pd


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


def split_period(
        start: Union[str, datetime],
        end: Union[str, datetime],
        validation_split_ratio: float,
        backtest_split_ratio: float,
        embargo_ratio: float,
        include_weekends: bool = False
) -> tuple:
    assert 0 < validation_split_ratio < 1
    assert 0 < backtest_split_ratio < 1
    assert 0 < embargo_ratio < 1
    assert validation_split_ratio + backtest_split_ratio + 2 * embargo_ratio < 1

    if isinstance(start, str):
        start = string_to_datetime(start)
    if isinstance(end, str):
        end = string_to_datetime(end)

    start_timestamp = start.timestamp()
    end_timestamp = end.timestamp()
    total_interval_length = end_timestamp - start_timestamp
    validation_split_length = validation_split_ratio * total_interval_length
    backtest_interval_length = backtest_split_ratio * total_interval_length
    train_interval_length = \
        (1 - validation_split_ratio - backtest_split_ratio - 2 * embargo_ratio) * total_interval_length
    embargo_offset = embargo_ratio * total_interval_length

    start_train_timestamp = start_timestamp
    end_train_timestamp = start_timestamp + train_interval_length
    start_train = datetime.fromtimestamp(start_train_timestamp)
    end_train = datetime.fromtimestamp(end_train_timestamp)

    start_validation_timestamp = end_train_timestamp + embargo_offset
    end_validation_timestamp = start_validation_timestamp + validation_split_length
    start_validation = datetime.fromtimestamp(start_validation_timestamp)
    end_validation = datetime.fromtimestamp(end_validation_timestamp)

    start_backtest_timestamp = end_validation_timestamp + embargo_offset
    end_backtest_timestamp = start_backtest_timestamp + backtest_interval_length
    start_backtest = datetime.fromtimestamp(start_backtest_timestamp)
    end_backtest = datetime.fromtimestamp(end_backtest_timestamp)

    assert start == start_train and end == end_backtest

    end_train = end_train.replace(hour=0, minute=0, second=0, microsecond=0)
    start_validation = start_validation.replace(hour=0, minute=0, second=0, microsecond=0)
    end_validation = end_validation.replace(hour=0, minute=0, second=0, microsecond=0)
    start_backtest = start_backtest.replace(hour=0, minute=0, second=0, microsecond=0)

    if not include_weekends:
        start_train = map_to_business_day(start_train)
        end_train = map_to_business_day(end_train)
        start_validation = map_to_business_day(start_validation)
        end_validation = map_to_business_day(end_validation)
        start_backtest = map_to_business_day(start_backtest)
        end_backtest = map_to_business_day(end_backtest)

    return (start_train, end_train), (start_validation, end_validation), (start_backtest, end_backtest)


def map_to_business_day(obj: Union[datetime, pd.Timestamp]) -> datetime:
    if isinstance(obj, datetime):
        obj = pd.Timestamp(obj)
    obj += 0 * pd.tseries.offsets.BDay()

    return obj.to_pydatetime()


def english_title_to_snake_case(string: str):
    return reduce(lambda x, y: x + ('_' if y == ' ' else y), string).lower()
