import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

from yacht import utils
from yacht.config import Config
from yacht.data.renderers import KFoldRenderer


class PurgedKFold(_BaseKFold):
    """
        Extend KFold to work with labels that span intervals.
        The train is purged of observations overlapping test-label intervals.
        Test set is assumed contiguous ( shuffle=False), w/0 training examples in between.
    """

    def __init__(
            self,
            start: datetime,
            end: datetime,
            interval: str,
            n_splits: int = 3,
            purge_ratio: float = 0.,
            embargo_ratio: float = 0.,
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)

        assert purge_ratio < 1
        assert embargo_ratio < 1

        self.start = start
        self.end = end
        self.interval = interval
        self.purge_ratio = purge_ratio
        self.embargo_ratio = embargo_ratio
        self.renderer = None

        self.from_to_series = self.build_from_to_series(start, end, interval)

        # Current state
        self.current_split = 0
        self.train_indices = None
        self.test_indices = None

    def split(self, X, y=None, groups=None):
        if (X.index == self.from_to_series.index).sum() != len(self.from_to_series):
            raise ValueError('X and date values must have the same index.')

        self.renderer = KFoldRenderer(prices=X)

        indices = np.arange(X.shape[0])
        embargo_offset = self.compute_embargo_offset(X)
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            test_start_index = self.from_to_series.index[i]  # Start of the test.
            test_indices = indices[i: j]
            test_end_index = self.from_to_series.index.searchsorted(self.from_to_series[test_indices].max())
            test_indices = self.apply_purge(test_indices)

            train_indices = self.from_to_series.index.searchsorted(
                self.from_to_series[self.from_to_series <= test_start_index].index
            )
            train_indices = self.apply_purge(train_indices)
            train_indices = np.concatenate([
                train_indices, indices[test_end_index + embargo_offset:]
            ])

            # Keep internal state
            self.current_split += 1
            self.train_indices = train_indices
            self.test_indices = test_indices

            yield train_indices, test_indices

    @classmethod
    def build_from_to_series(cls, start: datetime, end: datetime, interval: str) -> pd.Series:
        """
            Return: Index = start datetime of bar units, Values = end datetime of bar units -- for [start, end)
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

    def apply_purge(self, indices) -> np.array:
        return indices[:int(indices.shape[0] * (1 - self.purge_ratio))]

    def render(self, storage_dir, show=False):
        assert all((self.train_indices is not None, self.test_indices is not None))

        self.renderer.render(self.train_indices, self.test_indices)
        self.renderer.save(
            os.path.join(storage_dir, f'k_fold_split_{self.current_split}.png')
        )
        if show:
            self.renderer.show()

    def close(self):
        self.renderer.close()


#######################################################################################################################


def build_k_fold(config: Config) -> PurgedKFold:
    input_config = config.input
    train_config = config.train

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
        purge_ratio=train_config.k_fold_purge_ratio,
        embargo_ratio=train_config.k_fold_embargo_ratio
    )

    return k_fold
