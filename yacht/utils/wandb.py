import os
import warnings
from pathlib import Path

import wandb
from google.protobuf.json_format import MessageToDict, ParseDict
from stable_baselines3.common.callbacks import BaseCallback


from yacht import utils, Mode
from yacht.config import Config
from yacht.logger import Logger
from yacht.utils import create_project_name
from yacht.utils.cache import cache_experiment_tracker_name, CacheContext


class WandBContext(CacheContext):
    def __init__(self, config: Config, storage_dir: str):
        super().__init__(config=config, storage_dir=storage_dir)

        assert config.meta.experiment_tracker in ('', 'wandb'), \
            'If you are using the wandb context you should also set it from the config or at least leave it blank.'

        self.run_name = None
        self.run = None

    def __enter__(self):
        super().__enter__()

        if self.config.meta.experiment_tracker == 'wandb':
            cache_experiment_tracker_name(self.storage_dir, 'wandb')
            self._init_wandb()

        return self

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
        self._define_custom_step_metrics()

    def _define_custom_step_metrics(self):
        to_watch_metrics = set(self.config.meta.metrics_to_save_best_on). \
            union(set(self.config.meta.metrics_to_load_best_on))
        for mode in Mode:
            if not mode.is_trainable():
                wandb.define_metric(mode.to_step_key())
                wandb.define_metric(f'{mode.value}/*', step_metric=mode.to_step_key())

                for to_watch_metric in to_watch_metrics:
                    wandb.define_metric(f'{mode.value}/{to_watch_metric}', summary='max')
                    wandb.define_metric(f'{mode.value}/{to_watch_metric}', summary='min')
                    wandb.define_metric(f'{mode.value}/{to_watch_metric}', summary='mean')
                    wandb.define_metric(f'{mode.value}/{to_watch_metric}', summary='best')
        wandb.define_metric('timings_step')

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

        if self.config.meta.experiment_tracker == 'wandb':
            wandb.finish()

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


class HyperParameterTuningWandbContext(WandBContext):
    def _init_wandb(self):
        wandb.login(key=os.environ['WANDB_API_KEY'])

        self.run_name = create_project_name(self.config, self.storage_dir)
        self.run = wandb.init(
            project='yacht',
            entity='yacht'
        )
        self._define_custom_step_metrics()

    def get_config(self) -> Config:
        sweep_config = wandb.config._items
        del sweep_config['_wandb']
        sweep_config = self.split_keys(sweep_config)

        default_config = MessageToDict(self.config)
        sweep_config = utils.merge_configs(default_config, sweep_config)
        sweep_config = ParseDict(sweep_config, Config())

        return sweep_config

    @classmethod
    def split_keys(cls, config: dict) -> dict:
        """
        Args:
            config: Flattened config given by wandb sweeps YAML files.

        Returns:
            Nested dict used by the protobuf protocol.
        """

        new_config = dict()
        for k, v in config.items():
            possible_keys = k.split('.')
            current_key = possible_keys[0]
            rest_of_keys = '.'.join(possible_keys[1:])

            sub_config = {
                rest_of_keys: v
            }
            if len(possible_keys) == 1:
                new_config[k] = utils.convert_to_type(v)
            elif current_key in new_config:
                sub_config = cls.split_keys(sub_config)
                new_config[current_key] = utils.merge_configs(new_config[current_key], sub_config)
            else:
                new_config[current_key] = cls.split_keys(sub_config)

        return new_config


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
    def __init__(self, storage_dir: str, mode: Mode, verbose: int = 0):
        super().__init__(verbose)

        self.storage_dir = storage_dir
        self.mode = mode

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
            best_model_path = utils.build_best_reward_checkpoint_path(self.storage_dir, self.mode)
            if os.path.exists(best_model_path):
                wandb.save(best_model_path)
