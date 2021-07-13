from .paths import *
from .cache import *
from .misc import *
from .parsers import *

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from ..config import Config

logger_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN
}


def setup_logger(level: str, storage_dir: Optional[str] = None):
    Path(storage_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logger_levels[level])

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if storage_dir:
        file_handler = logging.FileHandler(os.path.join(storage_dir, 'logger.log'))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    log_dir = os.path.join(storage_dir, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

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
