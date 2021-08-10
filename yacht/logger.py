import logging
import os
from pathlib import Path
from typing import Optional

import colorlog
from stable_baselines3.common.logger import Logger as SB3Logger, INFO, DEBUG, WARN, ERROR, DISABLED

from yacht import utils
from yacht.utils import build_log_dir


class Logger(SB3Logger):
    logger_levels = {
        'info': INFO,
        'debug': DEBUG,
        'warn': WARN,
        'error': ERROR,
        'disabled': DISABLED
    }

    def __init__(self, storage_dir: str, level: str):
        if storage_dir:
            log_dir = build_log_dir(storage_dir)
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        else:
            log_dir = None

        # Set logger format & colour.
        log_format = (
            '%(asctime)s - '
            '%(name)s - '
            '%(funcName)s - '
            '%(levelname)s - '
            '%(message)s'
        )
        bold_seq = '\033[1m'
        colorlog_format = (
            f'{bold_seq} '
            '%(log_color)s '
            f'{log_format}'
        )
        colorlog.basicConfig(format=colorlog_format)

        self.logger = logging.getLogger()
        # The logging level will be controlled within this class.
        # So set the Python logger level to the lowest one that makes sense.
        self.logger.setLevel(logging.INFO)

        if storage_dir:
            formatter = logging.Formatter(log_format)

            # Set full logger.
            file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Set warning logger.
            file_handler = logging.FileHandler(os.path.join(log_dir, 'app.warning.log'))
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Set error logger
            file_handler = logging.FileHandler(os.path.join(log_dir, 'app.error.log'))
            file_handler.setLevel(logging.ERROR)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        super().__init__(
            folder=log_dir,
            output_formats=[]
        )

        # Set logging level.
        self.level_name = level
        self.set_level(self.logger_levels[level])

    def dump(self, step: int = 0) -> None:
        self.logger.debug(self.name_to_value)

    def _do_log(self, args) -> None:
        for arg in args:
            getattr(self.logger, self.level_name)(arg)


#######################################################################################################################


def build_logger(level: str, storage_dir: Optional[str] = None) -> Logger:
    from yacht.utils.wandb import WandBLogger

    if utils.get_experiment_tracker_name(storage_dir) == 'wandb':
        return WandBLogger(storage_dir=storage_dir, level=level)

    return Logger(storage_dir=storage_dir, level=level)
