import argparse
import logging

from agents import build_agent
from environment.environment import build_environment
from yacht.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", required=True, help='Path to your *.yaml configuration file.')
parser.add_argument("--logger-level", default='info', choices=('info', 'debug', 'warn'))


# TODO: Make a mechanism so data is not loaded 2x times for train & validation
# TODO: Add save model logic.
# TODO: See why btc cash is 0.

logger_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN
}

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(level=logger_levels[args.logger_level])

    config = Config(args.config_file)

    environment = build_environment(config=config)
    agent = build_agent(environment, config)
    agent.train()
