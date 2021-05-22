import os
from datetime import datetime, timedelta
from typing import Any, List, Dict

import pandas as pd

from binance.client import Client

from yacht.data.markets import Market


class Binance(Market):
    COLUMNS = (
        'Open',
        'High',
        'Low',
        'Close',
        'Volume'
    )

    def __init__(
            self,
            api_key,
            api_secret,
            storage_dir: str
    ):
        self.client = Client(api_key, api_secret)
        self.storage_file = os.path.join(storage_dir, 'binance.h5')

        super().__init__(api_key, api_secret, storage_dir)

    def open(self) -> pd.HDFStore:
        return pd.HDFStore(self.storage_file)

    def close(self):
        self.connection.close()

    def persist(self, interval: str):
        self.connection[interval].to_hdf(self.storage_file, interval, mode='w')

    def get(self, ticker: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
            Returns: data within [start, end)
        """

        if interval not in self.connection:
            raise RuntimeError(f'Table: "{interval}" not supported')
        end = end - timedelta(microseconds=1)

        database_to_pandas_freq = {
            'd': 'd',
            'h': 'h',
            'm': 'min'
        }
        freq = interval[:-1] + database_to_pandas_freq[interval[-1].lower()]
        date_time_index = pd.date_range(start=start, end=end, freq=freq)
        # Create the desired data span because there are missing values. In this way we will know exactly what data is
        # missing and at what index.
        final_data = pd.DataFrame(index=date_time_index, columns=self.COLUMNS)

        piece_of_data = self.connection[interval].loc[start:end]
        final_data.update(piece_of_data)

        final_data.fillna(method='bfill', inplace=True, axis=0)
        final_data.fillna(method='ffill', inplace=True, axis=0)

        return final_data

    def request(self, ticker: str, interval: str, start: datetime, end: datetime = None) -> List[List[Any]]:
        if '/' not in ticker:
            ticker = f'{ticker}USDT'
        else:
            ticker = ''.join(ticker.split('/'))

        start = start.strftime('%d %b, %Y')
        kwargs = {
            'symbol': ticker,
            'interval': interval,
            'start_str': start
        }
        if end:
            end = end.strftime('%d %b, %Y')
            kwargs['end_str'] = end

        return self.client.get_historical_klines(**kwargs)

    def process_request(self, data: List[List[Any]]) -> pd.DataFrame:
        df = pd.DataFrame(
            data,
            columns=[
                'Open time',
                'Open',
                'High',
                'Low',
                'Close',
                'Volume',
                'Close time',
                'Quote asset volume',
                'Number of trades',
                'Taker buy base asset volume',
                'Taker buy quote asset volume',
                'Ignore'
            ])

        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        df = df.loc[:, ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.set_index('Open time')

        return df

    def is_cached(self, interval: str, start: datetime, end: datetime) -> bool:
        if interval not in self.connection:
            return False

        return start in self.connection[interval].index and end and end in self.connection[interval].index

    def cache_request(self, interval: str, data: pd.DataFrame):
        if interval in self.connection:
            self.connection[interval] = self.connection[interval].combine_first(data)
        else:
            self.connection[interval] = data
