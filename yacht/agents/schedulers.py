from typing import Callable, Optional, Tuple

import pandas as pd

from yacht import utils


def constant_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Constant rate schedule.

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
        return initial_value

    return func


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear rate schedule.

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


def step_schedule(initial_value: float, drop_steps: Tuple[float] = (0.33, 0.66)) -> Callable[[float], float]:
    """
    Step rate schedule.

    :param initial_value: Initial learning rate.
    :param drop_steps: list with the progress points where the learning rate is dropped by a factor of 10
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    if 0.0 not in drop_steps:
        drop_steps = (0.0, *drop_steps)
    if 1.0 not in drop_steps:
        drop_steps = (*drop_steps, 1.0)

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

        if power is None:
            power = -len(intervals)

        return 10**power * initial_value

    return func


#######################################################################################################################


schedulers_registry = {
    'constant_schedule': constant_schedule,
    'linear_schedule': linear_schedule,
    'step_schedule': step_schedule
}


def build_scheduler(name: Optional[str], initial_value: float) -> Callable[[float], float]:
    if not name:
        name = 'constant_schedule'

    name = utils.camel_to_snake(name)
    scheduler = schedulers_registry[name]

    return scheduler(initial_value)
