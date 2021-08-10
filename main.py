import argparse
import logging
import os

import matplotlib
import wandb

from yacht.config import load_config, export_config
from yacht.data.datasets import build_dataset
from yacht.environments import build_env
from yacht import utils, Mode
from yacht import environments
from yacht.agents import build_agent
from yacht.evaluation import run_backtest
from yacht.evaluation.backtester import build_backtester
from yacht.trainer import build_trainer, run_train
from yacht.utils.wandb import WandBContext

logger = logging.getLogger(__file__)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('train', 'backtest', 'max_possible_profit'))
parser.add_argument(
    '--config_file_name',
    required=True,
    help='Name of the *.config file from the configuration dir.'
)
parser.add_argument('--resume_training', default=False, action='store_true', help='Resume training or not.')
parser.add_argument(
    '--agent_from',
    default='best',
    help='File path to the *.zip file that you want to resume from. If None it will resume from the best checkpoint.'
)
parser.add_argument('--storage_dir', required=True, help='Directory where your model & logs will be saved.')
parser.add_argument('--logger_level', default='info', choices=('info', 'debug', 'warn'))


if __name__ == '__main__':
    matplotlib.use('Agg')

    args = parser.parse_args()
    if '.' == args.storage_dir[0]:
        storage_dir = os.path.join(ROOT_DIR, args.storage_dir[2:])
    else:
        storage_dir = args.storage_dir

    utils.load_env(root_dir=ROOT_DIR)
    wandb.login(key=os.environ['WANDB_API_KEY'])
    log_dir = utils.setup_logger(
        level=args.logger_level,
        storage_dir=storage_dir
    )
    environments.register_gym_envs()
    config = load_config(utils.build_config_path(ROOT_DIR, args.config_file_name))
    export_config(config, storage_dir)
    logger.info(f'Config:\n{config}')

    with WandBContext(config, storage_dir):
        mode = Mode.from_string(args.mode)
        if mode == Mode.Train:
            run_train(
                config=config,
                storage_dir=storage_dir,
                resume_training=args.resume_training
            )

            if config.input.backtest.run:
                run_backtest(config=config, storage_dir=storage_dir, agent_from=args.agent_from)

        elif mode == Mode.Backtest:
            run_backtest(config=config, storage_dir=storage_dir, agent_from=args.agent_from)
