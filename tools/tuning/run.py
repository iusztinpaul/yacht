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
parser.add_argument('--logger_level', default='info', choices=('info', 'debug', 'warn'))


if __name__ == '__main__':
    matplotlib.use('Agg')

    args = parser.parse_args()
    if '.' == args.storage_dir[0]:
        storage_dir = os.path.join(ROOT_DIR, args.storage_dir[2:])
    else:
        storage_dir = args.storage_dir

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
            resume_training=False
        )
