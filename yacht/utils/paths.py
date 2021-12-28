import os
from pathlib import Path
from typing import Optional

from yacht import Mode


def adjust_relative_path(root_dir: str, relative_path: str) -> str:
    if '.' == relative_path[0]:
        return os.path.join(root_dir, relative_path[2:])
    else:
        return relative_path


def build_config_path(project_root_dir: str, config_name: str) -> str:
    return os.path.join(project_root_dir, 'yacht', 'config', 'configs', config_name)


def build_last_checkpoint_path(storage_dir: str, mode: Mode) -> str:
    return build_checkpoints_path(storage_dir, mode, 'last_model.zip')


def build_best_reward_checkpoint_path(storage_dir: str, mode: Mode) -> str:
    return build_checkpoints_path(storage_dir, mode, 'best_model.zip')


def build_checkpoints_dir(storage_dir: str, mode: Optional[Mode] = None) -> str:
    if mode is None:
        checkpoint_dir = os.path.join(storage_dir, 'checkpoints')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        return checkpoint_dir
    else:
        return build_checkpoints_path(storage_dir, mode, '')


def build_best_metric_checkpoint_path(storage_dir: str, mode: Mode, metric: str) -> str:
    return build_checkpoints_path(
        storage_dir,
        mode,
        build_best_metric_checkpoint_file_name(metric)
    )


def build_best_metric_checkpoint_file_name(metric: str) -> str:
    return f'best_model_{metric}.zip'


def build_checkpoints_path(storage_dir: str, mode: Mode, file_name: str) -> str:
    checkpoint_dir = os.path.join(storage_dir, 'checkpoints', mode.value)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    return os.path.join(checkpoint_dir, file_name)


def build_rewards_path(storage_dir: str, mode: Mode) -> str:
    return build_graphics_path(storage_dir, f'rewards_{mode.value}.png')


def build_graphics_path(storage_dir: str, file_name: str) -> str:
    graphics_dir = os.path.join(storage_dir, 'graphics')
    if not os.path.exists(graphics_dir):
        os.mkdir(graphics_dir)

    return os.path.join(graphics_dir, file_name)


def build_cache_path(storage_dir: str) -> str:
    return os.path.join(storage_dir, '.cache.json')


def build_monitor_dir(storage_dir: str, mode: Mode) -> str:
    return os.path.join(build_log_dir(storage_dir), mode.value)


def build_log_dir(storage_dir: str) -> str:
    return os.path.join(storage_dir, 'log')
