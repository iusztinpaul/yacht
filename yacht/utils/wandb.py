import logging
from pathlib import Path

import wandb
from google.protobuf.json_format import MessageToDict

from yacht import utils
from yacht.config import Config
from yacht.utils import create_project_name
from yacht.utils.cache import cache_experiment_tracker_name

logger = logging.getLogger(__file__)


class WandBContext:
    def __init__(self, config: Config, storage_dir):
        self.config = config
        self.storage_dir = storage_dir

        self.run_name = None
        self.run = None

    def __enter__(self):
        if self.config.meta.experiment_tracker == 'wandb':
            cache_experiment_tracker_name(self.storage_dir, 'wandb')
            self._init_wandb()

            logger.info(f'WandB run name: {self.run_name}')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.meta.experiment_tracker == 'wandb':
            wandb.finish()
            cache_experiment_tracker_name(self.storage_dir, '')

            logger.info(f'Finished run name: {self.run_name}')

    def _init_wandb(self):
        self.run_name = create_project_name(self.config, self.storage_dir)
        config = MessageToDict(self.config)

        self.run = wandb.init(
            project='yacht',
            entity='yacht',
            name=self.run_name,
            config=config
        )

    @classmethod
    def log_image_from(cls, path: str):
        save_dir = str(Path(path).parent.absolute())
        if 'graphics' in save_dir:
            save_dir = str(Path(save_dir).parent.absolute())

        if utils.get_experiment_tracker_name(save_dir) == 'wandb':
            file_name = utils.file_path_to_name(path)
            wandb.log({
                file_name: wandb.Image(path)
            })
