from abc import ABCMeta
from typing import List

import numpy as np
import pandas as pd


class BasePortfolio(metaclass=ABCMeta):
    def __init__(self, asset_names: List[str], time_span: int):
        """
        Args:
            asset_names: List of the asset names that are relevant to the algorithm.
            time_span: Time period over which the weights will be persisted.
        """
        self.asset_names = asset_names
        self.time_span = time_span

        self.portfolio_vector_memory = pd.DataFrame(
            index=pd.RangeIndex(time_span),
            columns=asset_names,
            dtype=np.float64
        )
        self.portfolio_vector_memory.fillna(1.0 / len(asset_names), inplace=True)

    def get_last_weights(self):
        return self.get_weights_at(-1)

    def get_weights_at(self, index: int):
        return self.portfolio_vector_memory.iloc[index]

    def set_last_weights(self, weights: np.array):
        self.set_weights_at(-1, weights)

    def set_weights_at(self, index: int, weights: np.array):
        if weights.shape[0] != len(self.asset_names):
            raise RuntimeError('Wrong number of weights distribution.')

        self.portfolio_vector_memory.iloc[index] = weights


class Portfolio(BasePortfolio):
    pass

