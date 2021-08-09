import pandas as pd
import numpy as np
from pyfolio import timeseries

from yacht import utils


def compute_backtest_metrics(report: pd.DataFrame, total_assets_col_name='total'):
    daily_return_report = get_daily_return(report, total_assets_col_name=total_assets_col_name)
    # TODO: See how to use positions & transactions parameters.
    strategy_metrics = timeseries.perf_stats(
        returns=daily_return_report,
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

    # Ignore hold actions ( action = 0) because their are irrelevant in this metric.
    positive_positions_mask = actions > 0
    negative_positions_mask = actions < 0

    statistics = dict()
    if positive_positions_mask.any():
        statistics['buy_pa'] = _compute_price_advantage(
            actions[positive_positions_mask],
            prices[positive_positions_mask],
            buy=True
        )

    if negative_positions_mask.any():
        statistics['sell_pa'] = _compute_price_advantage(
            actions[negative_positions_mask],
            prices[negative_positions_mask],
            buy=False
        )

    return statistics


def _compute_price_advantage(actions: np.ndarray, prices: np.ndarray, buy: bool = True) -> float:
    average_execution_price = (actions * prices).sum() / actions.sum()
    average_price = np.mean(prices)

    # If you buy, you want a lower AEP, else if you sell, you want a higher AEP.
    if buy:
        pa = 1 - average_execution_price / average_price
    else:
        pa = average_execution_price / average_price - 1
    pa *= 1e4

    return pa
