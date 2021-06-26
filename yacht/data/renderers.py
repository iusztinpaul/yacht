import datetime
import os
from abc import ABC
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from yacht.environments import Position
from yacht import utils


class BaseRenderer(ABC):
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

        self.fig = None
        self.ax = None

    def render(self, **kwargs):
        raise NotImplementedError()


class MatPlotLibRenderer(BaseRenderer, ABC):
    def show(self):
        if self.fig:
            self.fig.show()

    def close(self):
        if self.fig:
            plt.close(self.fig)

    def save(self, file_path: str):
        if self.fig:
            self.fig.savefig(file_path)


class MplFinanceRenderer(BaseRenderer, ABC):
    def render(self, save_file_path: str, **kwargs):
        raise NotImplementedError()


class KFoldRenderer(MatPlotLibRenderer):
    def __init__(self, prices: pd.DataFrame):
        super().__init__(prices)

        self.prices = self.prices.loc[:, 'Close']

    def render(self, **kwargs):
        train_indices: np.array = kwargs['train_indices']
        val_indices: np.array = kwargs['val_indices']

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


class TrainTestSplitRenderer(MatPlotLibRenderer):
    def __init__(
            self,
            prices: pd.DataFrame,
            train_interval: Tuple[datetime.datetime, datetime.datetime],
            test_interval: Tuple[datetime.datetime, datetime.datetime]
    ):
        super().__init__(prices)
        self.prices = self.prices.loc[:, 'Close']

        assert len(train_interval) == 2 and len(test_interval) == 2
        assert train_interval[0] < train_interval[1] and test_interval[0] < test_interval[1]

        self.train_interval = train_interval
        self.test_interval = test_interval

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


class TradingRenderer(MplFinanceRenderer):
    def __init__(self, prices: pd.DataFrame, start: datetime, end: datetime):
        super().__init__(prices)

        self.start = start
        self.end = end
        self.num_days = utils.get_num_days(start, end)

    def render(self, save_file_path: str, **kwargs):
        positions: List[Optional[Position]] = kwargs['positions']
        actions: List[int] = kwargs['actions']

        # Calculate positions.
        num_missing_positions = self.num_days - len(positions)
        positions = positions + [np.nan] * num_missing_positions
        positions: pd.Series = pd.Series(index=self.prices.index, data=positions)

        short_positions_index = positions[positions == Position.Short].index
        short_positions = positions.copy()
        short_positions[positions == Position.Short] = self.prices.loc[short_positions_index, 'High']
        short_positions[positions == Position.Long] = np.nan
        short_positions[positions == Position.Hold] = np.nan

        long_positions_index = positions[positions == Position.Long].index
        long_positions = positions.copy()
        long_positions[positions == Position.Long] = self.prices.loc[long_positions_index, 'Low']
        long_positions[positions == Position.Short] = np.nan
        long_positions[positions == Position.Hold] = np.nan

        hold_position_index = positions[positions == Position.Hold].index
        hold_positions = positions.copy()
        hold_positions[positions == Position.Hold] = \
            (self.prices.loc[hold_position_index, 'Low'] + self.prices.loc[hold_position_index, 'High']) / 2
        hold_positions[positions == Position.Short] = np.nan
        hold_positions[positions == Position.Long] = np.nan

        # Calculate actions.
        num_missing_actions = self.num_days - len(actions)
        actions = actions + [0] * num_missing_actions
        actions: pd.Series = pd.Series(index=self.prices.index, data=actions)

        additional_plots = [
            mpf.make_addplot(actions, panel=1, color='b', type='bar', width=1, ylabel='Actions')
        ]
        if len(short_positions[short_positions.notna()]) > 0:
            additional_plots.append(
                mpf.make_addplot(short_positions, type='scatter', markersize=25, marker='v', color='r')
            )
        if len(long_positions[long_positions.notna()]) > 0:
            additional_plots.append(
                mpf.make_addplot(long_positions, type='scatter', markersize=25, marker='^', color='g')
            )
        if len(hold_positions[hold_positions.notna()]) > 0:
            additional_plots.append(
                mpf.make_addplot(hold_positions, type='scatter', markersize=25, marker='.', color='y')
            )

        title = os.path.split(save_file_path)[1].split('.')[0]
        mpf.plot(
            self.prices,
            addplot=additional_plots,
            title=title,
            type='candle',
            ylabel='Prices',
            panel_ratios=(1, 1, 0.5),
            figratio=(2, 1),
            figscale=1.5,
            savefig=save_file_path,
            volume=True,
            volume_panel=2
        )
