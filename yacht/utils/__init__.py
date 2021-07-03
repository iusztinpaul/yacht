from .parsers import *
from .misc import *

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


logger_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN
}


def setup_logger(level: str, storage_path: Optional[str] = None):
    Path(storage_path).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logger_levels[level])

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if storage_path:
        file_handler = logging.FileHandler(os.path.join(storage_path, 'logger.log'))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    log_dir = os.path.join(storage_path, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    return log_dir


def load_env(root_dir: str):
    env_path = Path(root_dir) / '.env.default'
    load_dotenv(dotenv_path=env_path)

    env_path = Path(root_dir) / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
