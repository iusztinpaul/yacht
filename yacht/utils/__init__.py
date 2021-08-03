import colorlog

from .paths import *
from .cache import *
from .misc import *
from .parsers import *
from .sequence import *

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from yacht.config import Config

logger_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN
}


def setup_logger(level: str, storage_dir: Optional[str] = None):
    if storage_dir:
        log_dir = build_log_path(storage_dir)
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

    # Set logging level.
    root_logger = logging.getLogger()
    root_logger.setLevel(logger_levels[level])

    if storage_dir:
        formatter = logging.Formatter(log_format)

        # Set full logger.
        file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Set warning logger.
        file_handler = logging.FileHandler(os.path.join(log_dir, 'app.warning.log'))
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Set error logger
        file_handler = logging.FileHandler(os.path.join(log_dir, 'app.error.log'))
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return log_dir


def load_env(root_dir: str):
    env_path = Path(root_dir) / '.env.default'
    load_dotenv(dotenv_path=env_path)

    env_path = Path(root_dir) / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)


def create_project_name(config: Config, storage_dir: str):
    project_iteration = get_project_iteration(storage_dir)
    name = f'{config.environment.name}__{config.agent.name}__{project_iteration}'

    return name


def get_project_iteration(storage_dir: str, key: str = 'num_iteration') -> int:
    num_iteration = query_cache(storage_dir, key)
    if num_iteration is None:
        num_iteration = 0
    else:
        num_iteration += 1

    write_to_cache(storage_dir, key, num_iteration)

    return num_iteration
