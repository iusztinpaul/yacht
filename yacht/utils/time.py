import calendar
import time
from datetime import datetime, timedelta
from typing import List, Tuple

from dateutil.tz import tz

from entities import Frequency


def datetime_to_seconds(d: datetime, tz='utc') -> int:
    if tz == 'utc':
        return int(calendar.timegm(d.timetuple()))
    elif tz == 'local':
        return int(time.mktime(d.timetuple()))
    else:
        raise RuntimeError(f'Unsupported tz: {tz}')


def str_to_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d")


def compute_datetime_span(start: datetime, end: datetime, frequency: timedelta) -> List[datetime]:
    span = []
    current_datetime = start
    while current_datetime < end:
        span.append(current_datetime)

        current_datetime += frequency

    span.append(end)

    return span


def split_datetime_span(start_datetime: datetime, end_datetime: datetime, split: float, frequency: Frequency = None):
    assert split <= 1.

    start = datetime_to_seconds(start_datetime)
    end = datetime_to_seconds(end_datetime)
    span = end - start

    start_split_1 = start
    end_split_1 = start + span * (1 - split) - 1

    start_split_2 = start + span * (1 - split)
    end_split_2 = end

    start_split_1, end_split_1 = datetime.utcfromtimestamp(start_split_1), datetime.utcfromtimestamp(end_split_1)
    start_split_2, end_split_2 = datetime.utcfromtimestamp(start_split_2), datetime.utcfromtimestamp(end_split_2)

    if frequency:
        start_split_1 = round_datetime(start_split_1, frequency)
        end_split_1 = round_datetime(end_split_1, frequency)

        start_split_2 = round_datetime(start_split_2, frequency)
        end_split_2 = round_datetime(end_split_2, frequency)

    assert start_split_1 == start_datetime
    assert end_split_2 == end_datetime

    return (start_split_1, end_split_1), (start_split_2, end_split_2)


def round_datetime(dt: datetime, frequency: Frequency):
    # TODO: Give another look at this rounding
    dt = dt - timedelta(
        minutes=dt.minute % (frequency.seconds * 60),
        seconds=dt.second,
        microseconds=dt.microsecond
    )

    return dt
