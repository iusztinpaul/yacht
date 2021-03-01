import argparse

from agents import build_agent
from environment.environment import build_environment
from yacht.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", required=True, help='Path to your *.yaml configuration file.')


# TODO: Add train / test split logic.
# TODO: Finish agent.


if __name__ == '__main__':
    args = parser.parse_args()

    config = Config(args.config_file)

    environment = build_environment(config=config)
    agent = build_agent(environment, config)
