import datetime
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BaseRenderer:
    def __init__(self):
        self.fig = None
        self.ax = None

    def render(self, prices: pd.Series, *args, **kwargs):
        raise NotImplementedError()

    def show(self):
        self.fig.show()

    def save(self, file_path: str):
        self.fig.savefig(file_path)


class KFoldRenderer(BaseRenderer):
    def render(self, prices: pd.Series, train_indices: np.array, val_indices: np.array):
        self.fig, self.ax = plt.subplots()

        self.ax.plot(prices)
        self.ax.set_xticks(
            prices.index[
                np.linspace(0, prices.shape[0] - 1, 5).astype(np.int16)
            ])

        y_min = np.min(prices) * 0.9
        y_max = np.max(prices) * 1.1

        x_line_2_left = None
        x_line_2_right = None
        if np.max(val_indices) < np.min(train_indices):
            # Val split is in the left.
            x_line_1_left = prices.index[val_indices[-1]]
            x_line_1_right = prices.index[train_indices[0]]
        elif np.max(train_indices) < np.min(val_indices):
            # Val split is in the right.
            x_line_1_left = prices.index[train_indices[-1]]
            x_line_1_right = prices.index[val_indices[0]]
        else:
            # Val split is in the middle.
            diff = np.diff(train_indices)
            diff = np.concatenate([np.ones(shape=(1,)), diff])
            discontinuity_point = np.where(diff != 1)[0]
            assert len(discontinuity_point) == 1, 'There should be only one discontinuity point.'

            x_line_1_left = prices.index[train_indices[discontinuity_point - 1].item()]
            x_line_1_right = prices.index[val_indices[0]]
            x_line_2_left = prices.index[val_indices[-1]]
            x_line_2_right = prices.index[train_indices[discontinuity_point].item()]

        self.ax.text(
            x=prices.index[int(np.median(val_indices))],
            y=y_max + 10,
            s='Val'
        )
        self.ax.vlines(
            x_line_1_left,
            ymin=y_min,
            ymax=y_max,
            linestyles='dashed',
            colors='gray'
        )
        self.ax.vlines(
            x_line_1_right,
            ymin=y_min,
            ymax=y_max,
            linestyles='dashed',
            colors='gray'
        )
        if x_line_2_left and x_line_2_left:
            self.ax.vlines(
                x_line_2_left,
                ymin=y_min,
                ymax=y_max,
                linestyles='dashed',
                colors='gray'
            )
            self.ax.vlines(
                x_line_2_right,
                ymin=y_min,
                ymax=y_max,
                linestyles='dashed',
                colors='gray'
            )


class TrainTestSplitRenderer(BaseRenderer):
    def __init__(
            self,
            train_interval: Tuple[datetime.datetime, datetime.datetime],
            test_interval: Tuple[datetime.datetime, datetime.datetime]
    ):
        assert len(train_interval) == 2 and len(test_interval) == 2
        assert train_interval[0] < train_interval[1] and test_interval[0] < test_interval[1]

        self.train_interval = train_interval
        self.test_interval = test_interval

    def render(self, prices: pd.Series):
        self.fig, self.ax = plt.subplots()
        self.ax.plot(prices)

        y_min = np.min(prices) * 0.9
        y_max = np.max(prices) * 1.1

        if self.train_interval[1] < self.test_interval[0]:
            x_line_left = self.train_interval[1]
            x_line_right = self.test_interval[0]
        else:
            x_line_left = self.test_interval[1]
            x_line_right = self.train_interval[0]

        start = min(self.train_interval[0], self.test_interval[0])
        end = max(self.train_interval[1], self.test_interval[1])
        self.ax.set_xticks([
            start,
            (x_line_left - start) / 2 + start,
            x_line_left,
            (end - x_line_left) / 2 + x_line_left,
            end
        ])

        self.ax.text(
            x=(self.train_interval[1] - self.train_interval[0]) / 2.5 + self.train_interval[0],
            y=y_max + 10,
            s='TrainVal Split'
        )
        self.ax.text(
            x=(self.test_interval[1] - self.test_interval[0]) / 2.5 + self.test_interval[0],
            y=y_max + 10,
            s='Test Split'
        )
        self.ax.vlines(
            x_line_left,
            ymin=y_min,
            ymax=y_max,
            linestyles='dashed',
            colors='gray'
        )
        self.ax.vlines(
            x_line_right,
            ymin=y_min,
            ymax=y_max,
            linestyles='dashed',
            colors='gray'
        )


class PriceRenderer(BaseRenderer):
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def render(self):
        pass


class RewardRenderer(BaseRenderer):
    def render(self):
        pass


class ActionRenderer(BaseRenderer):
    def render(self):
        pass


class Renderer(BaseRenderer):
    price_renderer = PriceRenderer
    reward_renderer = RewardRenderer
    action_renderer = ActionRenderer

    def render(self):
        pass
