import datetime
import os
import sys
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common import results_plotter

from yacht import utils, Mode
from yacht.utils.wandb import WandBContext


class BaseRenderer(ABC):
    def __init__(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        self.data = data

        self.fig = None
        self.ax = None

    def render(self, **kwargs):
        self._render(**kwargs)

    @abstractmethod
    def _render(self, **kwargs):
        pass


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

            WandBContext.log_image_from(file_path)


class MplFinanceRenderer(BaseRenderer, ABC):
    def render(self, title: str, save_file_path: str, **kwargs):
        self._render(title, save_file_path, **kwargs)

        WandBContext.log_image_from(save_file_path)

    @abstractmethod
    def _render(self, title: str, save_file_path: str, **kwargs):
        pass


class KFoldRenderer(MatPlotLibRenderer):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

        self.prices = self.data.loc[:, 'Close']

    def _render(self, **kwargs):
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
            data: Dict[str, pd.DataFrame],
            train_split: Tuple[datetime.datetime, datetime.datetime],
            validation_split: Tuple[datetime.datetime, datetime.datetime],
            backtest_split: Tuple[datetime.datetime, datetime.datetime],
            rescale: bool = True
    ):
        super().__init__(data)

        assert len(train_split) == 2 and len(validation_split) == 2 and len(backtest_split) == 2
        assert train_split[0] < train_split[1] \
               < validation_split[0] < validation_split[1] \
               <= backtest_split[0] <= backtest_split[1]

        self.prices = self._get_prices(rescale)
        self.train_split = train_split
        self.validation_split = validation_split
        self.backtest_split = backtest_split
        self.has_backtest_split = (backtest_split[1] - backtest_split[0]) > datetime.timedelta(days=1)

    def _get_prices(self, rescale) -> Dict[str, pd.Series]:
        prices = dict()

        for ticker, values in self.data.items():
            if rescale:
                scaler = MinMaxScaler()

                indices = values.index
                values = values.loc[:, 'Close'].values.reshape(-1, 1)
                scaler.fit(values)

                values = scaler.transform(values)
                values = values.reshape(-1)
                prices[ticker] = pd.Series(index=indices, data=values)
            else:
                prices[ticker] = values.loc[:, 'Close']

        return prices

    def _render(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

        y_min = sys.float_info.max
        y_max = sys.float_info.min
        for ticker, prices in self.prices.items():
            self.ax.plot(prices, label=ticker)

            y_min = np.min(prices) if np.min(prices) < y_min else y_min
            y_max = np.max(prices) if np.max(prices) > y_max else y_max
        y_min *= 0.9
        y_max *= 1.1

        if self.has_backtest_split:
            splits = (
                    (self.train_split, 'Train'),
                    (self.validation_split, 'Validation'),
                    (self.backtest_split, 'Backtest')
            )
            self.ax.set_xticks([
                self.train_split[0],
                (self.validation_split[0] - self.train_split[1]) / 2 + self.train_split[1],
                (self.backtest_split[0] - self.validation_split[1]) / 2 + self.validation_split[1],
                self.backtest_split[1]
            ])
        else:
            splits = (
                (self.train_split, 'Train'),
                (self.validation_split, 'Validation')
            )
            self.ax.set_xticks([
                self.train_split[0],
                self.train_split[0] + (self.train_split[1] - self.train_split[0]) / 2,
                (self.validation_split[0] - self.train_split[1]) / 2 + self.train_split[1],
                self.validation_split[0] + (self.validation_split[1] - self.validation_split[0]) / 2,
                self.validation_split[1]
            ])
        for split, name in splits:
            self.ax.text(
                x=(split[1] - split[0]) / 4 + split[0],
                y=y_max + 10,
                s=name
            )
            self.ax.vlines(
                split[0],
                ymin=y_min,
                ymax=y_max,
                linestyles='dashed',
                colors='gray'
            )
            self.ax.vlines(
                split[1],
                ymin=y_min,
                ymax=y_max,
                linestyles='dashed',
                colors='gray'
            )
        self.fig.legend(bbox_to_anchor=(1., 1.))


class AssetEnvironmentRenderer(MplFinanceRenderer):
    COLOURS = (
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan'
    )

    def __init__(self, data: pd.DataFrame, start: datetime, end: datetime, unadjusted_start: datetime):
        super().__init__(data)

        self.start = start
        self.end = end
        self.unadjusted_start = unadjusted_start

    def _render(self, title: str, save_file_path: str, **kwargs):
        tickers: List[str] = kwargs['tickers']
        actions = kwargs['actions']
        total_cash = kwargs.get('total_cash')
        total_units = kwargs.get('total_units')
        total_assets = kwargs.get('total_assets')
        remove_adjacent_positions = kwargs.get('remove_adjacent_positions', False)
        render_positions_separately = kwargs.get('render_positions_separately', False)
        mean_price = kwargs.get('mean_price', None)

        assert len(tickers) <= len(self.COLOURS), 'Not enough supported colours.'

        if len(tickers) == 0:
            return

        # Cumsum actions & units for a better gradual visualization.
        # Split positive & negative actions for the cumsum operation.
        positive_actions = actions.copy()
        positive_actions[positive_actions < 0] = 0
        positive_actions = np.cumsum(positive_actions, axis=1)
        negative_actions = actions.copy()
        negative_actions[negative_actions > 0] = 0
        negative_actions = np.cumsum(negative_actions, axis=1)
        # Bring together all the actions.
        actions = positive_actions + negative_actions
        total_units = np.cumsum(total_units, axis=1)

        extra_plots = []
        legend_patches = []
        dummy_data = None
        for ticker_idx in range(len(tickers) - 1, -1, -1):
            ticker = tickers[ticker_idx]
            legend_patches.append(
                mpatches.Patch(color=self.COLOURS[ticker_idx], label=ticker)
            )

            ticker_actions: np.ndarray = actions[:, ticker_idx]
            # Trim extra data from the end.
            data = self.data.loc[(slice(None), ticker), ][:len(ticker_actions)]
            data = data.reset_index(level=1, drop=True)
            # Keep a reference only for the indices for the main plot.
            dummy_data = data
            extra_plots.append(
                mpf.make_addplot(
                    data['Close'],
                    panel=0,
                    color=self.COLOURS[ticker_idx],
                    secondary_y=False
                )
            )

            if render_positions_separately:
                trading_renderer = TradingPositionRenderer(
                    data=data,
                    unadjusted_start=self.unadjusted_start,
                    remove_adjacent_positions=remove_adjacent_positions
                )
                filename, file_extension = os.path.splitext(save_file_path)
                trading_renderer.render(
                    title=f'{ticker}',
                    save_file_path=f'{filename}_{ticker}{file_extension}',
                    actions=ticker_actions
                )
            else:
                extra_plots.extend(
                    TradingPositionRenderer.compute_positions_plots(
                        actions=ticker_actions,
                        data=data,
                        remove_adjacent_positions=remove_adjacent_positions
                    )
                )

            # Calculate actions.
            ticker_actions: pd.Series = pd.Series(index=data.index, data=ticker_actions)
            extra_plots.append(
                mpf.make_addplot(
                    ticker_actions,
                    panel=1,
                    type='bar',
                    width=1,
                    ylabel='Actions',
                    color=self.COLOURS[ticker_idx],
                    secondary_y=False
                ),
            )

            # Calculate units.
            ticker_units = total_units[:, ticker_idx]
            ticker_units: pd.Series = pd.Series(index=data.index, data=ticker_units)
            extra_plots.append(
                mpf.make_addplot(
                    ticker_units,
                    panel=2,
                    type='bar',
                    width=1,
                    ylabel='Units',
                    color=self.COLOURS[ticker_idx],
                    secondary_y=False
                ),
            )

        total_cash = pd.Series(index=dummy_data.index, data=total_cash)
        total_assets = pd.Series(index=dummy_data.index, data=total_assets)
        extra_plots.extend([
            mpf.make_addplot(total_cash, panel=3, color='tab:gray', type='bar', width=1, ylabel='Cash'),
            mpf.make_addplot(total_assets, panel=4, color='tab:gray', type='bar', width=1, ylabel='Assets')
        ])

        vlines = dict(
            vlines=[self.unadjusted_start],
            linewidths=(1.5, ),
            linestyle='-.'
        )
        if mean_price is not None:
            hlines = dict(
                hlines=mean_price.values.tolist(),
                colors=[self.COLOURS[idx] for idx in range(len(tickers) - 1, -1, -1)],
                linewidths=[1.0 for _ in range(len(tickers))],
                linestyle='-.'
            )
        else:
            hlines = None
        fig, axes = mpf.plot(
            data=dummy_data,
            addplot=extra_plots,
            title=title,
            type='line',
            ylabel=f'Prices',
            panel_ratios=(1, 0.75, 0.75, 0.75, 0.75),
            figratio=(2, 1),
            figscale=1.5,
            savefig=save_file_path,
            volume=False,
            axisoff=False,
            returnfig=True,
            vlines=vlines,
            hlines=hlines
        )

        # Configure chart legend and title
        fig.legend(handles=legend_patches)
        # Save figure to file
        fig.savefig(save_file_path)


class TradingPositionRenderer(MplFinanceRenderer):
    def __init__(
            self,
            data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
            unadjusted_start: datetime,
            remove_adjacent_positions: bool
    ):
        super().__init__(data)

        self.unadjusted_start = unadjusted_start
        self.remove_adjacent_positions = remove_adjacent_positions

    def _render(self, title: str, save_file_path: str, **kwargs):
        actions: np.ndarray = kwargs['actions']
        # Remove extra data.
        self.data = self.data.iloc[:len(actions)]

        # Calculate positions.
        additional_plots = self.compute_positions_plots(
            actions=actions,
            data=self.data,
            remove_adjacent_positions=self.remove_adjacent_positions
        )

        # Calculate actions.
        actions: pd.Series = pd.Series(index=self.data.index, data=actions)
        additional_plots.append(
            mpf.make_addplot(
                actions,
                panel=1,
                type='bar',
                width=1,
                ylabel='Actions',
                color='tab:blue',
            ),
        )

        vlines = dict(
            vlines=[self.unadjusted_start],
            linewidths=(1.5,),
            linestyle='-.'
        )
        mpf.plot(
            self.data,
            addplot=additional_plots,
            title=title,
            type='line',
            ylabel='Prices',
            figratio=(2, 1),
            panel_ratios=(1, 0.75),
            figscale=1.5,
            savefig=save_file_path,
            volume=False,
            vlines=vlines
        )

    @classmethod
    def compute_positions_plots(
            cls,
            actions: np.ndarray,
            data: pd.DataFrame,
            remove_adjacent_positions: bool = False
    ) -> list:
        # We are interested only in the direction of the action.
        positions = np.sign(actions)

        # Replace with nan adjacent values that are equal.
        if remove_adjacent_positions:
            adjacent_equal_values_mask = np.zeros(positions.shape[0], dtype=bool)
            adjacent_equal_values_mask[1:] = positions[1:] == positions[:-1]
            positions[adjacent_equal_values_mask] = np.nan

        # Calculate positions.
        positions = pd.Series(index=data.index, data=positions, dtype=np.float32)
        # Precompute masks.
        positive_positions_mask = positions > 0
        hold_positions_mask = positions == 0
        negative_positions_mask = positions < 0

        short_positions_index = positions[negative_positions_mask].index
        short_positions = pd.Series(index=positions.index, dtype=np.float32)
        short_positions[negative_positions_mask] = data.loc[short_positions_index, 'High']
        short_positions[positions >= 0] = np.nan

        long_positions_index = positions[positive_positions_mask].index
        long_positions = pd.Series(index=positions.index, dtype=np.float32)
        long_positions[positive_positions_mask] = data.loc[long_positions_index, 'Low']
        long_positions[positions <= 0] = np.nan

        hold_position_index = positions[hold_positions_mask].index
        hold_positions = pd.Series(index=positions.index, dtype=np.float32)
        hold_positions[hold_positions_mask] = \
            (data.loc[hold_position_index, 'Low'] + data.loc[hold_position_index, 'High']) / 2
        hold_positions[negative_positions_mask] = np.nan
        hold_positions[positive_positions_mask] = np.nan

        additional_plots = []
        if len(short_positions[short_positions.notna()]) > 0:
            additional_plots.append(
                mpf.make_addplot(short_positions, type='scatter', markersize=20, marker='v', color='r')
            )
        if len(long_positions[long_positions.notna()]) > 0:
            additional_plots.append(
                mpf.make_addplot(long_positions, type='scatter', markersize=20, marker='^', color='g')
            )
        if len(hold_positions[hold_positions.notna()]) > 0:
            additional_plots.append(
                mpf.make_addplot(hold_positions, type='scatter', markersize=20, marker='.', color='y')
            )

        return additional_plots


class RewardsRenderer(MatPlotLibRenderer):
    def __init__(self, total_timesteps: int, storage_dir: str, mode: Mode):
        super().__init__(None)

        self.total_timesteps = total_timesteps
        self.storage_dir = storage_dir
        self.mode = mode

    def _render(self):
        log_dir = utils.build_monitor_dir(self.storage_dir, mode=self.mode)
        results_plotter.plot_results(
            dirs=[log_dir],
            num_timesteps=self.total_timesteps,
            x_axis=results_plotter.X_TIMESTEPS,
            task_name=f'rewards_{self.mode.value}'
        )

    def show(self):
        super().show()

        plt.show()

    def close(self):
        super().close()

        plt.close()

    def save(self, file_path: str):
        super().save(file_path)

        plt.savefig(file_path)
        WandBContext.log_image_from(file_path)
