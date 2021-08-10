import os
from pathlib import Path

import wandb
from google.protobuf.json_format import MessageToDict
from stable_baselines3.common.callbacks import BaseCallback


from yacht import utils
from yacht.config import Config
from yacht.logger import Logger
from yacht.utils import create_project_name
from yacht.utils.cache import cache_experiment_tracker_name


class WandBContext:
    def __init__(self, config: Config, storage_dir):
        assert config.meta.experiment_tracker in ('', 'wandb'), \
            'If you are using the wandb context you should either turn it on or off.'

        self.config = config
        self.storage_dir = storage_dir

        self.run_name = None
        self.run = None

    def __enter__(self):
        # Clear possible residuals from last runs.
        cache_experiment_tracker_name(self.storage_dir, '')

        if self.config.meta.experiment_tracker == 'wandb':
            cache_experiment_tracker_name(self.storage_dir, 'wandb')
            self._init_wandb()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cache_experiment_tracker_name(self.storage_dir, '')

        if self.config.meta.experiment_tracker == 'wandb':
            wandb.finish()

    def _init_wandb(self):
        wandb.login(key=os.environ['WANDB_API_KEY'])

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


class WandBLogger(Logger):
    def dump(self, step: int = 0) -> None:
        super().dump(step)

        wandb.log(self.name_to_value)

    def _do_log(self, args) -> None:
        super()._do_log(args)

        for arg in args:
            if isinstance(arg, dict):
                wandb.log(arg)


class WandBCallback(BaseCallback):
    def __init__(self, storage_dir: str, verbose: int = 0):
        super().__init__(verbose)

        self.storage_dir = storage_dir

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        policy = self.locals['self'].policy

        wandb.watch(
            (
                policy.features_extractor,
                policy.mlp_extractor,
                policy.value_net,
                policy.action_net
            ),
            log='parameters',
            log_freq=100
        )

    def _on_training_end(self) -> None:
        policy = self.locals['self'].policy

        wandb.unwatch(
            (
                policy.features_extractor,
                policy.mlp_extractor,
                policy.value_net,
                policy.action_net
            )
        )

        if utils.get_experiment_tracker_name(self.storage_dir) == 'wandb':
            best_model_path = utils.build_best_checkpoint_path(self.storage_dir)
            if os.path.exists(best_model_path):
                wandb.save(best_model_path)