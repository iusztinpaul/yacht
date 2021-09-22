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


STORAGE_DIR = os.path.join(ROOT_DIR, 'storage', 'hyper_param_optimization', f'{uuid1()}')
CONFIG_DIR = utils.build_config_path(ROOT_DIR, 'single_asset_order_execution_ppo.config.txt')


if __name__ == '__main__':
    matplotlib.use('Agg')

    utils.load_env_variables(root_dir=ROOT_DIR)
    environments.register_gym_envs()
    logger = yacht.logger.build_logger(
        level='info',
        storage_dir=STORAGE_DIR
    )
    config = load_config(CONFIG_DIR)

    with HyperParameterTuningWandbContext(config, STORAGE_DIR) as context:
        config = context.get_config()
        logger.info(f'Config:\n{config}')

        run_train(
            config=config,
            logger=logger,
            storage_dir=STORAGE_DIR,
            resume_training=False
        )
