from typing import List, Any


def compute_max_score(num_days: int, action_max_score: int):
    return num_days * action_max_score


def fib_sequence(n: int) -> List[int]:
    def _sequence(n: int) -> int:
        if n == 0:
            return 1
        elif n == 1:
            return 1
        else:
            return _sequence(n-1) + _sequence(n-2)

    values = []
    for _n in range(n):
        values.append(_sequence(_n))

    return values


def convert_to_type(obj: str) -> Any:
    if is_number(obj):
        return float(obj)
    if not isinstance(obj, str):
        return obj
    if obj.upper() == 'TRUE':
        return True
    if obj.upper() == 'FALSE':
        return False

    return obj


def is_number(obj: Any) -> bool:
    return isinstance(obj, (int, float))
