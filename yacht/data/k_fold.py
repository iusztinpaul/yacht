from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

from yacht import utils
from yacht.config import InputConfig, TrainConfig


class PurgedKFold(_BaseKFold):
    """
        Extend KFold to work with labels that span intervals.
        The train is purged of observations overlapping test-label intervals.
        Test set is assumed contiguous ( shuffle=False), w/0 training examples in between.
    """

    def __init__(self, start: datetime, end: datetime, interval: str, n_splits: int = 3, embargo_ratio: float = 0.):
        super().__init__(n_splits, shuffle=False, random_state=None)

        assert embargo_ratio < 1

        self.start = start
        self.end = end
        self.interval = interval
        self.embargo_ratio = embargo_ratio

        self.from_to_series = self.build_from_to_series(start, end, interval)

    def split(self, X, y=None, groups=None):
        if (X.index == self.from_to_series.index).sum() != len(self.from_to_series):
            raise ValueError('X and date values must have the same index.')

        indices = np.arange(X.shape[0])
        embargo_offset = self.compute_embargo_offset(X)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            test_start_index = self.from_to_series.index[i]  # Start of the test.
            test_indices = indices[i: j]
            test_end_index = self.from_to_series.index.searchsorted(self.from_to_series[test_indices].max())
            train_indices = self.from_to_series.index.searchsorted(
                self.from_to_series[self.from_to_series <= test_start_index].index
            )
            train_indices = np.concatenate([
                train_indices, indices[test_end_index + embargo_offset:]
            ])

            yield train_indices, test_indices

    @classmethod
    def build_from_to_series(cls, start: datetime, end: datetime, interval: str) -> pd.Series:
        """
            Return: Index = start of bar units, Values = end of bar units -- for [start, end)
        """

        timedelta = utils.interval_to_timedelta(interval)

        current_index_value = start
        index_list = [start]
        # TODO: Is it close or opened interval ?
        while current_index_value < end - timedelta:
            current_index_value += timedelta
            index_list.append(current_index_value)

        values_list = index_list[1:]
        values_list.append(index_list[-1] + timedelta)

        from_to_series = pd.Series(
            data=values_list,
            index=index_list
        )

        return from_to_series

    def compute_embargo_offset(self, X) -> int:
        return int(X.shape[0] * self.embargo_ratio)


#######################################################################################################################


def build_k_fold(input_config: InputConfig, train_config: TrainConfig) -> PurgedKFold:
    train_val_start, train_val_end, _, _ = utils.split_period(
        input_config.start,
        input_config.end,
        input_config.back_test_split_ratio,
        train_config.k_fold_embargo_ratio
    )

    k_fold = PurgedKFold(
        start=train_val_start,
        end=train_val_end,
        interval='1d',
        n_splits=train_config.k_fold_splits,
        embargo_ratio=train_config.k_fold_embargo_ratio
    )

    return k_fold
