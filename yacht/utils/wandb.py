import logging

import wandb
from google.protobuf.json_format import MessageToDict

from yacht.config import Config
from yacht.utils import create_project_name


logger = logging.getLogger(__file__)


def init_wandb(config: Config, storage_dir: str):
    name = create_project_name(config, storage_dir)
    config = MessageToDict(config)

    wandb.init(
        project='yacht',
        entity='iusztinpaul',
        name=name,
        config=config
    )

    return name


class WandBContext:
    def __init__(self, config: Config, storage_dir):
        self.config = config
        self.storage_dir = storage_dir
        self.run_name = None

    def __enter__(self):
        self.run_name = init_wandb(self.config, self.storage_dir)

        logger.info(f'WandB run name: {self.run_name}')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()

        logger.info(f'Finished run name: {self.run_name}')
