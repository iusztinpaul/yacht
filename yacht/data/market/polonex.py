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

    COINS = ['ETH', 'LTC', 'XRP', 'ETC', 'DASH', 'XMR', 'XEM', 'ZEC', 'DCR', 'STR']
    REQUESTED_FEATURES = ['close', 'high', 'low']

    @property
    def commission(self) -> float:
        return 0.0025

    @property
    def tickers(self) -> List[str]:
        return self.COINS

    @property
    def features(self) -> List[str]:
        return self.REQUESTED_FEATURES

    def get_all(self, start: datetime, end: datetime):
        data_frequency = self.input_config.data_frequency.seconds
        start = utils.datetime_to_seconds(start)
        end = utils.datetime_to_seconds(end)

        time_index = pd.to_datetime(
            list(range(
                start,
                end,
                data_frequency
            )),
            unit='s'
        )
        coins_history = pd.DataFrame(
            columns=self.REQUESTED_FEATURES,
            index=pd.MultiIndex.from_product(
                (self.COINS, time_index),
                names=('coin', 'datetime')
            )
        )

        connection = sqlite3.connect(self.DATABASE_NAME)
        try:
            for row_number, coin in enumerate(self.COINS):
                for feature in self.REQUESTED_FEATURES:
                    # NOTE: transform the start date to end date
                    if feature == "close":
                        sql = (
                            "SELECT date+300 AS date_norm, close FROM History WHERE"
                            " date_norm>={start} and date_norm<{end}"
                            " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                                start=start, end=end, period=data_frequency, coin=coin
                            )
                        )
                    elif feature == "open":
                        sql = (
                            "SELECT date+{period} AS date_norm, open FROM History WHERE"
                            " date_norm>={start} and date_norm<{end}"
                            " and date_norm%{period}=0 and coin=\"{coin}\"".format(
                                start=start, end=end, period=data_frequency, coin=coin
                            )
                        )
                    elif feature == "volume":
                        sql = (
                                "SELECT date_norm, SUM(volume)" +
                                " FROM (SELECT date+{period}-(date%{period}) "
                                "AS date_norm, volume, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<{end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=data_frequency, start=start, end=end, coin=coin
                                )
                        )
                    elif feature == "high":
                        sql = (
                                "SELECT date_norm, MAX(high)" +
                                " FROM (SELECT date+{period}-(date%{period})"
                                " AS date_norm, high, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<{end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=data_frequency, start=start, end=end, coin=coin
                                )
                        )
                    elif feature == "low":
                        sql = (
                                "SELECT date_norm, MIN(low)" +
                                " FROM (SELECT date+{period}-(date%{period})"
                                " AS date_norm, low, coin FROM History)"
                                " WHERE date_norm>={start} and date_norm<{end} and coin=\"{coin}\""
                                " GROUP BY date_norm".format(
                                    period=data_frequency, start=start, end=end, coin=coin
                                )
                        )
                    else:
                        raise ValueError(f"The feature {feature} is not supported")

                    serial_data = pd.read_sql_query(
                        sql,
                        con=connection,
                        parse_dates=["date_norm"],
                        index_col="date_norm"
                    )
                    coin_feature_data = serial_data.squeeze().astype(np.float32).values
                    coins_history.loc[(coin, serial_data.index), feature] = coin_feature_data
        finally:
            connection.commit()
            connection.close()

        coins_history = coins_history.fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
        # TODO: This is a trick because there is no data at all for bfill or ffill. Is it ok ?
        coins_history = coins_history.fillna(1e10-27)

        # data_matrix = features x coins x (window_size + 1)
        # M = batch_size x features x coins x (window_size + 1)
        # X = batch_size x features x coins x window_size
        # y = batch_size x features x coins

        coin_num = len(coins_history.index.get_level_values('coin').unique())
        datetime_num = len(coins_history.index.get_level_values('datetime').unique())
        features_num = coins_history.shape[1]

        coins_history = coins_history.values
        coins_history = np.transpose(coins_history, (1, 0))
        coins_history = coins_history.reshape((features_num, coin_num, datetime_num))

        # TODO: It would be nice to visualize 'coins_history' at this point. This point is very crucial & error prone.

        return coins_history

    def get(self, start: datetime, end: datetime, ticker: str) -> np.array:
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
