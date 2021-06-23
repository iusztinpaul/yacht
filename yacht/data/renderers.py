import datetime
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from yacht.environments import Position
from yacht import utils


class BaseRenderer:
    def __init__(self, prices: pd.Series):
        self.prices = prices

        self.fig = None
        self.ax = None

    def render(self, *args, **kwargs):
        raise NotImplementedError()

    def show(self):
        if self.fig:
            self.fig.show()

    def close(self):
        if self.fig:
            plt.close(self.fig)

    def save(self, file_path: str):
        if self.fig:
            self.fig.savefig(file_path)


class KFoldRenderer(BaseRenderer):
    def render(self, train_indices: np.array, val_indices: np.array):
        self.fig, self.ax = plt.subplots()

        self.ax.plot(self.prices)
        self.ax.set_xticks(
            self.prices.index[
                np.linspace(0, self.prices.shape[0] - 1, 5).astype(np.int16)
            ])

        y_min = np.min(self.prices) * 0.9
        y_max = np.max(self.prices) * 1.1

        x_line_2_left = None
        x_line_2_right = None
        if np.max(val_indices) < np.min(train_indices):
            # Val split is in the left.
            x_line_1_left = self.prices.index[val_indices[-1]]
            x_line_1_right = self.prices.index[train_indices[0]]
        elif np.max(train_indices) < np.min(val_indices):
            # Val split is in the right.
            x_line_1_left = self.prices.index[train_indices[-1]]
            x_line_1_right = self.prices.index[val_indices[0]]
        else:
            # Val split is in the middle.
            diff = np.diff(train_indices)
            diff = np.concatenate([np.ones(shape=(1,)), diff])
            discontinuity_point = np.where(diff != 1)[0]
            assert len(discontinuity_point) == 1, 'There should be only one discontinuity point.'

            x_line_1_left = self.prices.index[train_indices[discontinuity_point - 1].item()]
            x_line_1_right = self.prices.index[val_indices[0]]
            x_line_2_left = self.prices.index[val_indices[-1]]
            x_line_2_right = self.prices.index[train_indices[discontinuity_point].item()]

        self.ax.text(
            x=self.prices.index[int(np.median(val_indices))],
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
            prices: pd.Series,
            train_interval: Tuple[datetime.datetime, datetime.datetime],
            test_interval: Tuple[datetime.datetime, datetime.datetime]
    ):
        super().__init__(prices)

        assert len(train_interval) == 2 and len(test_interval) == 2
        assert train_interval[0] < train_interval[1] and test_interval[0] < test_interval[1]

        self.train_interval = train_interval
        self.test_interval = test_interval

    # def render(self, file_path: str = None):
    #     if self.train_interval[1] < self.test_interval[0]:
    #         x_line_left = self.train_interval[1]
    #         x_line_right = self.test_interval[0]
    #     else:
    #         x_line_left = self.test_interval[1]
    #         x_line_right = self.train_interval[0]
    #
    #     plot_arguments = {
    #         'data': self.prices,
    #         'type': 'line',
    #         'vlines': {
    #             'vlines': [x_line_left, x_line_right],
    #             'colors': 'g',
    #             'linestyle': '-.',
    #             'linewidths': 1.5
    #         }
    #     }
    #     if file_path:
    #         plot_arguments['savefig'] = {
    #             'fname': file_path,
    #             'dpi': 100,
    #         }
    #
    #     mpf.plot(**plot_arguments)

    def render(self):
        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.prices)

        y_min = np.min(self.prices) * 0.9
        y_max = np.max(self.prices) * 1.1

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


class TradingRenderer(BaseRenderer):
    def __init__(self, prices: pd.Series, start: datetime, end: datetime):
        super().__init__(prices)

        self.start = start
        self.end = end
        self.num_days = utils.get_num_days(start, end)

        self.rendered_prices = None

    def render(self, positions: List[Optional[Position]]):
        plt.cla()

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xticks(
                self.prices.index[
                    np.linspace(0, self.prices.shape[0] - 1, 5).astype(np.int16)
                ])

        self.ax.plot(self.prices)
        self.ax.plot(self.prices[:len(positions)])

        num_missing_positions = self.num_days - len(positions)
        positions = positions + [None] * num_missing_positions

        position_ticks = pd.Series(index=self.prices.index)
        position_history = np.array(positions)
        position_ticks[position_history == Position.Short] = Position.Short
        position_ticks[position_history == Position.Long] = Position.Long

        short_positions = position_ticks[position_ticks == Position.Short]
        self.ax.plot(
            short_positions.index,
            self.prices.loc[short_positions.index],
            'rv',
            markersize=6
        )

        long_positions = position_ticks[position_ticks == Position.Long]
        self.ax.plot(
            long_positions.index,
            self.prices.loc[long_positions.index],
            'g^',
            markersize=6
        )

    def pause(self):
        plt.pause(0.0005)
