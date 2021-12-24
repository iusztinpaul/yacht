import argparse
import os

import matplotlib

import yacht.logger
from yacht.config import load_config, export_config
from yacht import utils, Mode
from yacht import environments
from yacht.data.datasets import build_tickers
from yacht.data.markets import build_market
from yacht.evaluation import run_backtest
from yacht.trainer import run_train
from yacht.utils.wandb import WandBContext


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('train', 'backtest', 'download', 'export_actions'))
parser.add_argument(
    '--config_file_name',
    required=True,
    help='Name of the *.config file from the configuration dir.'
)
parser.add_argument('--resume_training', default=False, action='store_true', help='Resume training or not.')
parser.add_argument(
    '--agent_from',
    default=None,
    help='File path to the *.zip file that you want to resume from.'
)
parser.add_argument('--storage_dir', required=True, help='Directory where your model & logs will be saved.')
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
    export_config(config, storage_dir)

    with WandBContext(config, storage_dir) as e:
        logger = yacht.logger.build_logger(
            level=args.logger_level,
            storage_dir=storage_dir
        )
        logger.info(f'Config:\n{config}')

        mode = Mode.from_string(args.mode)
        if mode == Mode.Train:
            run_train(
                config=config,
                logger=logger,
                storage_dir=storage_dir,
                resume_training=args.resume_training,
                agent_from=args.agent_from,
                market_storage_dir=market_storage_dir
            )
            if config.input.backtest.run:
                run_backtest(
                    config=config,
                    logger=logger,
                    storage_dir=storage_dir,
                    agent_from=args.agent_from,
                    mode=Mode.BestMetricBacktestTest,
                    market_storage_dir=market_storage_dir
                )

        elif mode.is_backtestable():
            run_backtest(
                config=config,
                logger=logger,
                storage_dir=storage_dir,
                agent_from=args.agent_from,
                mode=Mode.BestMetricBacktestTrain,
                market_storage_dir=market_storage_dir
            )
            run_backtest(
                config=config,
                logger=logger,
                storage_dir=storage_dir,
                agent_from=args.agent_from,
                mode=Mode.BestMetricBacktestValidation,
                market_storage_dir=market_storage_dir
            )
            run_backtest(
                config=config,
                logger=logger,
                storage_dir=storage_dir,
                agent_from=args.agent_from,
                mode=Mode.BestMetricBacktestTest,
                market_storage_dir=market_storage_dir
            )
        elif mode == Mode.Download:
            storage_dir = market_storage_dir if market_storage_dir is not None else storage_dir
            logger.info(f'Downloading to: {storage_dir}')

            market = build_market(
                config=config,
                logger=logger,
                storage_dir=storage_dir,
                read_only=False
            )
            tickers = build_tickers(config, mode)
            for interval in config.input.intervals:
                market.download(
                    tickers,
                    interval=interval,
                    start=utils.string_to_datetime(config.input.start),
                    end=utils.string_to_datetime(config.input.end),
                    squeeze=True,
                    config=config
                )

            logger.info(f'Downloading finished')
        elif mode == Mode.ExportActions:
            run_backtest(
                config=config,
                logger=logger,
                storage_dir=storage_dir,
                agent_from=args.agent_from,
                mode=Mode.BacktestTrain,
                market_storage_dir=market_storage_dir
            )
