import argparse

import numpy as np

from data.market.polonex import PoloniexMarket
from yacht.configs import Config
from yacht.environment import Portfolio

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", required=True, help='Path to your *.yaml configuration file.')


if __name__ == '__main__':
    args = parser.parse_args()

    config = Config(args.config_file)
    portfolio = Portfolio(tickers=['AAPL', 'MSFT'], time_span=config.input_config.data_span_seconds)

    market = PoloniexMarket()
    market_data = market.request_coins(
        config.input_config.start_date,
        config.input_config.end_date,
    )
    print(market_data)