import argparse
import os
import sys
from pathlib import Path
from uuid import uuid1

import matplotlib

ROOT_DIR = str(Path(os.path.abspath(__file__)).parent.parent.parent)
sys.path.append(ROOT_DIR)

import yacht.logger
from yacht import utils
from yacht import environments
from yacht.config import load_config
from yacht.trainer import run_train
from yacht.utils.wandb import HyperParameterTuningWandbContext


parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file_name',
    required=True,
    help='Name of the *.config file from the configuration dir.'
)
parser.add_argument(
    '--storage_dir',
    default=os.path.join('storage', 'hyper_param_optimization', f'{uuid1()}'),
    help='Directory where your model & logs will be saved.'
)
parser.add_argument(
    '--market_storage_dir',
    default=None,
    help='Optional directory where you want to save your dataset. If not specified it will be saved in "--storage_dir".'
         'If this parameter is specified than the market is read only for parallel trainings.'
)
parser.add_argument('--logger_level', default='info', choices=('info', 'debug', 'warn'))


if __name__ == '__main__':
    matplotlib.use('Agg')

    args = parser.parse_args()
    storage_dir = utils.adjust_relative_path(ROOT_DIR, args.storage_dir)
    if args.market_storage_dir is not None:
        market_storage_dir = utils.adjust_relative_path(ROOT_DIR, args.market_storage_dir)
    else:
        market_storage_dir = None

    utils.load_env_variables(root_dir=ROOT_DIR)
    environments.register_gym_envs()
    config = load_config(utils.build_config_path(ROOT_DIR, args.config_file_name))

    with HyperParameterTuningWandbContext(config, storage_dir) as context:
        logger = yacht.logger.build_logger(
            level=args.logger_level,
            storage_dir=storage_dir
        )

        logger.info(f'Loading default config: {args.config_file_name}')
        config = context.get_config()
        logger.info(f'Config:\n{config}')

        run_train(
            config=config,
            logger=logger,
            storage_dir=storage_dir,
            resume_training=False,
            market_storage_dir=market_storage_dir
        )
