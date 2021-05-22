from datetime import datetime, timedelta
from typing import Union


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


def get_num_days(start: Union[str, datetime], end: Union[str, datetime]) -> int:
    if isinstance(start, str):
        start = string_to_datetime(start)
    if isinstance(end, str):
        end = string_to_datetime(end)

    difference = end - start

    return difference.days


def split_period(start: datetime, end: datetime, split_ratio: float, offset_ratio: float = 0.01):
    assert split_ratio < 1
    assert offset_ratio < 1

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
