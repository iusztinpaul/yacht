import json
import logging
import sqlite3
from typing import List

import numpy as np
import pandas as pd
from datetime import datetime

from urllib.request import Request, urlopen
from urllib.parse import urlencode

import utils
from config import InputConfig
from data.market import BaseMarket

logger = logging.getLogger(__file__)


class PoloniexMarket(BaseMarket):
    DATABASE_NAME = 'polonex.db'
    PUBLIC_COMMANDS = [
        'returnTicker',
        'return24hVolume',
        'returnOrderBook',
        'returnTradeHistory',
        'returnChartData',
        'returnCurrencies',
        'returnLoanOrders'
    ]

    # COINS = sorted(['ETH', 'LTC', 'XRP', 'ETC', 'DASH', 'XMR', 'XEM', 'ZEC', 'DCR', 'STR'])
    COINS = sorted(['ETH', 'LTC', 'XRP', 'ETC', 'DASH', 'XMR', 'XEM', 'DCR', 'STR'])
    REQUESTED_FEATURES = ['close', 'high', 'low']

    def __init__(self, input_config: InputConfig):
        super().__init__(input_config)

        self.time_index = pd.to_datetime(
            list(range(
                self.input_config.start_datetime_seconds,
                self.input_config.end_datetime_seconds + 1,
                self.input_config.data_frequency.seconds
            )),
            unit='s'
        )
        self.coins_history = pd.DataFrame(
            columns=self.REQUESTED_FEATURES,
            index=pd.MultiIndex.from_product(
                (self.COINS, self.time_index),
                names=('coin', 'datetime')
            )
        )

    @property
    def assets(self) -> pd.DataFrame:
        return self.coins_history

    @property
    def commission(self) -> float:
        return 0.0025

    @property
    def tickers(self) -> List[str]:
        return self.COINS

    @property
    def features(self) -> List[str]:
        return self.REQUESTED_FEATURES

    def load_all(self):
        data_frequency = self.input_config.data_frequency.seconds
        start = self.input_config.start_datetime_seconds
        end = self.input_config.end_datetime_seconds

        connection = sqlite3.connect(self.DATABASE_NAME)
        try:
            for row_number, coin in enumerate(self.COINS):
                coin_multi_index = pd.MultiIndex.from_product(
                    ((coin, ), self.time_index),
                    names=('coin', 'datetime')
                )
                for feature in self.REQUESTED_FEATURES:
                    if feature == "close":
                        sql = (
                            "SELECT coin, date+300 AS date_norm, close FROM History WHERE"
                            " date_norm>={start} and date_norm<={end}"
                            " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                                start=start, end=end, period=data_frequency, coin=coin
                            )
                        )
                    elif feature == "open":
                        sql = (
                            "SELECT coin, date+{period} AS date_norm, open FROM History WHERE"
                            " date_norm>={start} and date_norm<={end}"
                            " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                                start=start, end=end, period=data_frequency, coin=coin
                            )
                        )
                    elif feature == "volume":
                        sql = (
                                "SELECT coin, date_norm, SUM(volume)" +
                                " FROM (SELECT date+{period}-(date%{period}) "
                                "AS date_norm, volume, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=data_frequency, start=start, end=end, coin=coin
                                )
                        )
                    elif feature == "high":
                        sql = (
                                "SELECT coin, date_norm, MAX(high)" +
                                " FROM (SELECT date+{period}-(date%{period})"
                                " AS date_norm, high, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=data_frequency, start=start, end=end, coin=coin
                                )
                        )
                    elif feature == "low":
                        sql = (
                                "SELECT coin, date_norm, MIN(low)" +
                                " FROM (SELECT date+{period}-(date%{period})"
                                " AS date_norm, low, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<={end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=data_frequency, start=start, end=end, coin=coin
                                )
                        )
                    else:
                        raise ValueError(f"The feature {feature} is not supported")

                    feature_data = pd.read_sql_query(
                        sql,
                        con=connection,
                        parse_dates=['date_norm'],
                        index_col=['coin', 'date_norm']
                    )
                    feature_data.columns = (feature, )
                    feature_data = feature_data.reindex(coin_multi_index, fill_value=np.nan)
                    feature_data.fillna(inplace=True, method='bfill', axis=0)
                    feature_data.fillna(inplace=True, method='ffill', axis=0)

                    self.coins_history.update(
                        feature_data
                    )
        finally:
            connection.commit()
            connection.close()

    def get_all(self, start_dt: datetime, end_dt: datetime):
        data_frequency = self.input_config.data_frequency.seconds
        start = utils.datetime_to_seconds(start_dt)
        end = utils.datetime_to_seconds(end_dt)

        datetime_range = pd.to_datetime(
            list(range(
                start,
                end + 1,
                data_frequency
            )),
            unit='s'
        )
        coins_history = self.load_cached_data(datetime_range)

        coin_num = len(coins_history.index.get_level_values('coin').unique())
        datetime_num = len(coins_history.index.get_level_values('datetime').unique())
        features_num = coins_history.shape[1]

        coins_history = coins_history.values
        coins_history = np.transpose(coins_history, (1, 0))
        coins_history = coins_history.reshape((features_num, coin_num, datetime_num))

        # TODO: It would be nice to visualize 'coins_history' at this point. This point is very crucial & error prone.

        return coins_history

    def load_cached_data(self, datetime_range: pd.DatetimeIndex) -> pd.DataFrame:
        return self.coins_history.loc[
            (
                slice(None),
                datetime_range
             ),
        ].sort_index()

    def get(self, start_dt: datetime, end_dt: datetime, ticker: str) -> np.array:
        # TODO: Implement this ?
        raise NotImplementedError()

    def download(self, start: datetime, end: datetime):
        coin_requested_data = self.request_coins(start, end)

        connection = sqlite3.connect(self.DATABASE_NAME)
        try:
            cursor = connection.cursor()

            cursor.execute('CREATE TABLE IF NOT EXISTS History (date INTEGER,'
                           'coin varchar(20), high FLOAT, low FLOAT,'
                           'open FLOAT, close FLOAT, volume FLOAT, '
                           'quoteVolume FLOAT,'
                           'PRIMARY KEY (date, coin));')
            connection.commit()

            for coin, coin_data in coin_requested_data:
                for one_piece_of_data in coin_data:
                    cursor.execute(
                        'INSERT OR IGNORE INTO History VALUES (?,?,?,?,?,?,?,?);',
                        (
                            one_piece_of_data['date'],
                            coin,
                            one_piece_of_data['low'],
                            one_piece_of_data['high'],
                            one_piece_of_data['open'],
                            one_piece_of_data['close'],
                            one_piece_of_data['volume'],
                            one_piece_of_data['quoteVolume']
                        )
                    )

        finally:
            connection.commit()
            connection.close()

    def request_coins(self, start: datetime, end: datetime):
        all_data = []
        for coin in self.COINS:
            data = self.request_coin(start, end, coin)
            all_data.append(data)

        return list(zip(self.COINS, all_data))

    def request_coin(self, start: datetime, end: datetime, ticker: str):
        start = start.timestamp()
        end = end.timestamp()
        pair = f'BTC_{ticker}'

        return self.request_until_success(
            'returnChartData',
            {
                'currencyPair': pair,
                'period': 300,
                'start': start,
                'end': end
            }
        )

    def request_until_success(self, command, args):
        is_connect_success = False
        chart = {}
        while not is_connect_success:
            try:
                chart = self.api(command, args)
                is_connect_success = True
            except Exception as e:
                logger.error(e)

        return chart

    def api(self, command, args):
        """
        returns 'False' if invalid command or if no APIKey or Secret is specified (if command is "private")
        returns {"error":"<error message>"} if API error
        """
        if command in self.PUBLIC_COMMANDS:
            url = 'https://poloniex.com/public?'
            args['command'] = command
            ret = urlopen(Request(url + urlencode(args)))
            return json.loads(ret.read().decode(encoding='UTF-8'))
        else:
            return False
