import os
import sys



sys.path.append('/home/iusztin/Documents/projects/yacht')

import matplotlib

import yacht.logger
from yacht import utils, Mode
from yacht import environments
from yacht.config import load_config
from yacht.evaluation import run_backtest
from yacht.trainer import run_train
from yacht.utils.wandb import HyperParameterTuningWandbContext

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(ROOT_DIR, 'storage', 'hyper_param_optimization')
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
        config = context.wandb_to_proto_config()
        logger.info(f'Config:\n{config}')

        run_train(
            config=config,
            logger=logger,
            storage_dir=STORAGE_DIR,
            resume_training=False
        )
        # run_backtest(
        #     config=config,
        #     logger=logger,
        #     storage_dir=STORAGE_DIR,
        #     agent_from='best',
        #     mode=Mode.BacktestTrain
        # )
        #
        # if config.input.backtest.run:
        #     run_backtest(
        #         config=config,
        #         logger=logger,
        #         storage_dir=STORAGE_DIR,
        #         agent_from='best',
        #         mode=Mode.Backtest
        #     )
