import logging
from datetime import datetime
from typing import Any, List

import pandas as pd

from binance.client import Client
from binance.exceptions import BinanceAPIException

from yacht.data.markets.base import H5Market
from yacht.data.markets.mixins import TechnicalIndicatorMixin


logger = logging.getLogger(__file__)


class Binance(H5Market):
    def __init__(
            self,
            features: List[str],
            api_key,
            api_secret,
            storage_dir: str,
            include_weekends: bool
    ):
        super().__init__(features, api_key, api_secret, storage_dir, 'binance.h5', include_weekends)

        self.client = Client(api_key, api_secret)

    def request(self, ticker: str, interval: str, start: datetime, end: datetime = None) -> List[List[Any]]:
        if '-' not in ticker:
            ticker = f'{ticker}USDT'
        else:
            ticker = ''.join(ticker.split('-'))

        start = start.strftime('%d %b, %Y')
        kwargs = {
            'symbol': ticker,
            'interval': interval,
            'start_str': start
        }
        if end:
            end = end.strftime('%d %b, %Y')
            kwargs['end_str'] = end

        try:
            return self.client.get_historical_klines(**kwargs)
        except BinanceAPIException as e:
            logger.info(f'Binance does not support ticker: {ticker}')
            raise e

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
        df['Open time'] = df['Open time']
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        df = df.loc[:, ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.set_index('Open time')

        return df


class TechnicalIndicatorBinance(TechnicalIndicatorMixin, Binance):
    pass
