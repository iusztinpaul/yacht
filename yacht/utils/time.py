import calendar
import time
from datetime import datetime


def datetime_to_seconds(d: datetime, tz='utc') -> int:
    if tz == 'utc':
        return int(calendar.timegm(d.timetuple()))
    elif tz == 'local':
        return int(time.mktime(d.timetuple()))
    else:
        raise RuntimeError(f'Unsupported tz: {tz}')


def str_to_datetime(s: str) -> datetime:
    return datetime.strptime(s, "%Y/%m/%d")
