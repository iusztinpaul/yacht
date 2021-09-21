import os
from pathlib import Path

import wandb
from google.protobuf.json_format import MessageToDict, ParseDict
from stable_baselines3.common.callbacks import BaseCallback


from yacht import utils, Mode
from yacht.config import Config
from yacht.logger import Logger
from yacht.utils import create_project_name
from yacht.utils.cache import cache_experiment_tracker_name


class WandBContext:
    def __init__(self, config: Config, storage_dir: str):
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        cache_experiment_tracker_name(self.storage_dir, '')

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
            entity='yacht',
            name=self.run_name
        )

    def wandb_to_proto_config(self) -> Config:
        config = wandb.config._items
        del config['_wandb']
        config = self.split_keys(config)
        default_config = MessageToDict(self.config)
        config = self.update_nested_dict(default_config, config)
        config = ParseDict(config, Config())

        return config

    @classmethod
    def split_keys(cls, config: dict) -> dict:
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
                new_config[current_key].update(cls.split_keys(sub_config))
            else:
                new_config[current_key] = cls.split_keys(sub_config)

        return new_config

    @classmethod
    def update_nested_dict(cls, main_dict: dict, extra_dict: dict) -> dict:
        for new_k, new_v in extra_dict.items():
            if new_k not in main_dict:
                main_dict[new_k] = new_v
            elif not isinstance(new_v, dict):
                main_dict[new_k] = new_v
            else:
                main_dict[new_k] = cls.update_nested_dict(main_dict[new_k], extra_dict[new_k])

        return main_dict


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
