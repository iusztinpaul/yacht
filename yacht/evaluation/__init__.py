from yacht import Mode
from yacht.evaluation.backtester import build_backtester
from yacht.evaluation.metrics import *
from yacht.config import Config
from yacht.logger import Logger


def run_backtest(config: Config, logger: Logger, storage_dir: str, agent_from: str):
    logger.info('Starting backtesting...')

    train_backtester = build_backtester(
        config=config,
        logger=logger,
        storage_dir=storage_dir,
        mode=Mode.BacktestTrain,
        agent_from=agent_from
    )
    train_backtester.test()

    test_backtester = build_backtester(
        config=config,
        logger=logger,
        storage_dir=storage_dir,
        mode=Mode.Backtest,
        agent_from=agent_from
    )
    test_backtester.test()
