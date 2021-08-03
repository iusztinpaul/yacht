import pandas as pd


def get_daily_return(report: pd.DataFrame, value_col_name='total'):
    df = report.copy(deep=True)
    df[df == 0.] = 1e-2

    df['daily_return'] = df[value_col_name].pct_change(1)

    return pd.Series(df['daily_return'], index=df.index)
