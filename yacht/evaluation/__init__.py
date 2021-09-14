from yacht import Mode
from yacht.evaluation.backtester import build_backtester
from yacht.evaluation.metrics import *
from yacht.config import Config
from yacht.logger import Logger


def run_backtest(config: Config, logger: Logger, storage_dir: str, agent_from: str, mode: Mode):
    logger.info(f'Backtesting on {mode}')

    if mode == Mode.BacktestTrain:
        backtester_ = build_backtester(
            config=config,
            logger=logger,
            storage_dir=storage_dir,
            mode=Mode.BacktestTrain,
            agent_from=agent_from
        )
    elif mode == Mode.BacktestValidation:
        backtester_ = build_backtester(
            config=config,
            logger=logger,
            storage_dir=storage_dir,
            mode=Mode.BacktestValidation,
            agent_from=agent_from
        )
    elif mode == Mode.Backtest:
        backtester_ = build_backtester(
            config=config,
            logger=logger,
            storage_dir=storage_dir,
            mode=Mode.Backtest,
            agent_from=agent_from
        )
    else:
        raise RuntimeError(f'Wrong mode for backtesting: {mode}')

    if backtester_ is not None:
        backtester_.test()
        backtester_.close()
    else:
        logger.info(f'Backtester for mode: {mode.value} is not valid.')
