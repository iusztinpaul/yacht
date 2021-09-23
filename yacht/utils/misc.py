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
    if not isinstance(obj, str):
        return obj
    if is_number(obj):
        return float(obj)
    if obj.upper() == 'TRUE':
        return True
    if obj.upper() == 'FALSE':
        return False

    return obj


def is_number(obj: Any) -> bool:
    return isinstance(obj, (int, float))


def merge_configs(default_dict: dict, overriding_dict: dict) -> dict:
    """
    Args:
        default_dict:  The 'dict' config with the default values to be overwritten.
        overriding_dict: The 'dict' config with the new values. If it has some missing keys the default values
            from the default_dict will be kept.

    Returns:
        The merged 'dict' config with the overwritten values.
    """

    for overriding_key, overriding_value in overriding_dict.items():
        if overriding_key not in default_dict:
            default_dict[overriding_key] = overriding_value
        elif not isinstance(overriding_value, dict):
            default_dict[overriding_key] = overriding_value
        else:
            default_dict[overriding_key] = merge_configs(
                default_dict[overriding_key],
                overriding_dict[overriding_key]
            )

    return default_dict
