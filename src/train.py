import argparse

import numpy as np

from src.configs import Config
from src.environment import Portfolio

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", required=True, help='Path to your *.yaml configuration file.')


if __name__ == '__main__':
    args = parser.parse_args()

    config = Config(args.config_file)
    portfolio = Portfolio(asset_names=['AAPL', 'MSFT'], time_span=config.data_time_span)
