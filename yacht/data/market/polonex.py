import json
import logging
import time
import numpy as np
from datetime import datetime

from urllib.request import Request, urlopen
from urllib.parse import urlencode

from data.market import BaseMarket

logger = logging.getLogger(__file__)


class PoloniexMarket(BaseMarket):
    PUBLIC_COMMANDS = [
        'returnTicker',
        'return24hVolume',
        'returnOrderBook',
        'returnTradeHistory',
        'returnChartData',
        'returnCurrencies',
        'returnLoanOrders'
    ]

    COINS = ['ETH', 'LTC', 'XRP']

    def request_coins(self, start: datetime, end: datetime):
        requested_features = ('close', 'high', 'low')

        all_data = []
        for coin in self.COINS:
            data = self.get(start, end, coin)
            coin_data = [
                [unit_data[feature] for feature in requested_features]
                for unit_data in data
            ]
            all_data.append(coin_data)

        all_data = np.array(all_data, dtype=np.float32)

        return all_data

    def get(self, start: datetime, end: datetime, ticker: str):
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
