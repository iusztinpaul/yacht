import argparse
import shutil
from pathlib import Path

import utils
from agents import build_train_agent
from environments.environment import build_environment
from yacht2.config import Config

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True, help='Path to your *.yaml configuration file.')
parser.add_argument("--resume_training", default=False, action='store_true', help='Resume training or not.')
parser.add_argument("--storage_path", required=True, help='Path to the directory where your model & logs will be saved.')
parser.add_argument("--logger_level", default='info', choices=('info', 'debug', 'warn'))


# TODO: Implement back testing.
# TODO: See why btc cash is 0.


if __name__ == '__main__':
    args = parser.parse_args()
    Path(args.storage_path).mkdir(parents=True, exist_ok=True)
    utils.setup_logger(
        level=args.logger_level,
        storage_path=args.storage_path
    )

    config = Config(args.config_file)
    shutil.copy(args.config_file, args.storage_path)

    environment = build_environment(config=config, reason='train')
    agent = build_train_agent(environment, config, args.storage_path, args.resume_training)
    agent.train()
