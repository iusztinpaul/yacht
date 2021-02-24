import time
from datetime import datetime


def datetime_to_seconds(d: datetime) -> int:
    return int(time.mktime(d.timetuple()))


def str_to_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d")
