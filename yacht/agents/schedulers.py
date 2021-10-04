from typing import List, Callable

import pandas as pd


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def step_schedule(initial_value: float, drop_steps: List[float]) -> Callable[[float], float]:
    """
    Step rate schedule.

    :param initial_value: Initial learning rate.
    :param drop_steps: list with the progress points where the learning rate is dropped by a factor of 10
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    if 0.0 not in drop_steps:
        drop_steps.insert(0, 0.0)

    intervals = []
    for i in range(len(drop_steps) - 1):
        intervals.append(
            pd.Interval(left=drop_steps[i], right=drop_steps[i+1], closed='both')
        )

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        power = None
        for i, interval in enumerate(intervals):
            query = 1. - progress_remaining
            if query in interval:
                power = -i
                break

        return 10**power * initial_value

    return func
