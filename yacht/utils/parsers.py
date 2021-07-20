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
        split_ratio: float,
        offset_ratio: float
):
    assert split_ratio < 1
    assert offset_ratio < 1

    if isinstance(start, str):
        start = string_to_datetime(start)
    if isinstance(end, str):
        end = string_to_datetime(end)

    start_timestamp = start.timestamp()
    end_timestamp = end.timestamp()
    interval_length = end_timestamp - start_timestamp
    interval_1_length = (1 - split_ratio) * interval_length
    interval_2_length = split_ratio * interval_length

    start_1 = datetime.fromtimestamp(start_timestamp)
    end_1 = datetime.fromtimestamp(start_timestamp + interval_1_length)

    offset = interval_2_length * offset_ratio
    start_2 = datetime.fromtimestamp(start_timestamp + interval_1_length + offset)
    end_2 = datetime.fromtimestamp(start_timestamp + interval_1_length + interval_2_length)

    assert start == start_1 and end == end_2

    end_1 = end_1.replace(hour=0, minute=0, second=0, microsecond=0)
    start_2 = start_2.replace(hour=0, minute=0, second=0, microsecond=0)

    return start_1, end_1, start_2, end_2


def english_title_to_snake_case(string: str):
    return reduce(lambda x, y: x + ('_' if y == ' ' else y), string).lower()
