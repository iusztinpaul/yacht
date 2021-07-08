import json

import wandb
from google.protobuf.json_format import MessageToDict

from .misc import *
from .parsers import *
from .paths import *

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


def get_project_iteration(storage_dir: str) -> int:
    # TODO: For more cache data build some generic cache functions to query and create.
    cache_file_path = build_cache_path(storage_dir)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as f:
            local_cache = json.load(f)
    else:
        local_cache = dict()

    key = f'num_iteration'
    if key not in local_cache:
        local_cache[key] = 0
    else:
        local_cache[key] += 1

    with open(cache_file_path, 'w') as f:
        json.dump(local_cache, f)

    return local_cache[key]
