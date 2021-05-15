from datetime import datetime
from typing import Union


def string_to_datetime(string: str) -> datetime:
    # day/month/year
    return datetime.strptime(string, "%d/%m/%Y")


def get_num_days(start: Union[str, datetime], end: Union[str, datetime]) -> int:
    if isinstance(start, str):
        start = string_to_datetime(start)
    if isinstance(end, str):
        end = string_to_datetime(end)

    difference = end - start

    return difference.days
