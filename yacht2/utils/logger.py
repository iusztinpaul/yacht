import logging
import os
from typing import Optional

logger_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN
}


def setup_logger(level: str, storage_path: Optional[str] = None):
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
