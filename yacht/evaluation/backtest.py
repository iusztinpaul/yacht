import pandas as pd
from pyfolio import timeseries


def get_daily_return(report: pd.DataFrame, value_col_name='Total Value'):
    df = report.copy(deep=True)
    df[df == 0.] = 1e-2

    df['daily_return'] = df[value_col_name].pct_change(1)

    return pd.Series(df['daily_return'], index=df.index)


def compute_backtest_results(report: pd.DataFrame, value_col_name='Total Value'):
    daily_return_report = get_daily_return(report, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=daily_return_report,
        positions=None,
        transactions=None,
        turnover_denom='AGB',
    )

    final_report = pd.concat([report, daily_return_report], axis=1)

    return perf_stats_all, final_report
