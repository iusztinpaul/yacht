import pandas as pd
from pyfolio import timeseries

from yacht.utils.sequence import get_daily_return


def compute_backtest_results(report: pd.DataFrame, value_col_name='total'):
    daily_return_report = get_daily_return(report, value_col_name=value_col_name)
    # TODO: See how to use positions & transactions parameters.
    perf_stats_all = timeseries.perf_stats(
        returns=daily_return_report,
        positions=None,
        transactions=None,
        turnover_denom='AGB',
    )

    final_report = pd.concat([report, daily_return_report], axis=1)

    return perf_stats_all, final_report
