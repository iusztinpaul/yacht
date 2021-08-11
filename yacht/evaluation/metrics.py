import pandas as pd
import numpy as np
from pyfolio import timeseries

from yacht import utils


def compute_backtest_metrics(report: pd.DataFrame, total_assets_col_name='total'):
    daily_return_report = get_daily_return(report, total_assets_col_name=total_assets_col_name)

    # TODO: Add the rest of the arguments for more statistics.
    strategy_metrics = timeseries.perf_stats(
        returns=daily_return_report,
        factor_returns=None,
        positions=None,
        transactions=None,
        turnover_denom='AGB',
    )
    strategy_metrics = dict(
        zip(strategy_metrics.index, strategy_metrics.values)
    )

    # Map all indices from plain english title to snake case for consistency.
    snake_case_strategy_metrics = dict()
    for k, v in strategy_metrics.items():
        snake_case_strategy_metrics[utils.english_title_to_snake_case(k)] = v
    strategy_metrics = snake_case_strategy_metrics

    price_advantage_metrics = compute_price_advantage(report)
    strategy_metrics.update(price_advantage_metrics)

    longs_shorts_ratio = compute_longs_shorts_ratio(report)
    strategy_metrics['LSR'] = longs_shorts_ratio

    final_report = pd.concat([report, daily_return_report], axis=1)

    return strategy_metrics, final_report


def get_daily_return(report: pd.DataFrame, total_assets_col_name='total'):
    df = report.copy(deep=True)
    df[df == 0.] = 1e-2

    df['daily_return'] = df[total_assets_col_name].pct_change(1)

    return pd.Series(df['daily_return'], index=df.index)


def compute_price_advantage(report: pd.DataFrame) -> dict:
    actions = report.action.values
    prices = report.price.values
    mean_price = np.mean(prices)

    # Ignore hold actions ( action = 0) because their are irrelevant in this metric.
    positive_positions_mask = actions > 0
    negative_positions_mask = actions < 0

    statistics = dict()
    if positive_positions_mask.any():
        statistics['buy_pa'] = _compute_price_advantage(
            actions[positive_positions_mask],
            prices[positive_positions_mask],
            mean_price=mean_price,
            buy=True
        )

    if negative_positions_mask.any():
        statistics['sell_pa'] = _compute_price_advantage(
            actions[negative_positions_mask],
            prices[negative_positions_mask],
            mean_price=mean_price,
            buy=False
        )

    return statistics


def _compute_price_advantage(
        actions: np.ndarray,
        prices: np.ndarray,
        mean_price: np.ndarray,
        buy: bool = True
) -> float:
    average_execution_price = (actions * prices).sum() / actions.sum()

    # If you buy, you want a lower AEP, else if you sell, you want a higher AEP.
    if buy:
        pa = 1 - average_execution_price / mean_price
    else:
        pa = average_execution_price / mean_price - 1
    pa *= 1e4

    return pa


def compute_longs_shorts_ratio(report: pd.DataFrame) -> float:
    final_num_longs = report.longs.values[-1]
    final_num_shorts = report.shorts.values[-1]

    longs_shorts_ratio = final_num_longs / (final_num_shorts + 1e-17)

    return longs_shorts_ratio.item()
