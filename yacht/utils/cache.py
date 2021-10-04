import json
import os
from pathlib import Path
from typing import Any

from yacht.config import Config
from yacht.utils import build_cache_path


def cache_experiment_tracker_name(storage_dir: str, experiment_tracker_name: str):
    CacheContext.write_to_cache(storage_dir, 'experiment_tracker_name', experiment_tracker_name)


def get_experiment_tracker_name(storage_dir: str) -> str:
    return CacheContext.query_cache(storage_dir, 'experiment_tracker_name')


def get_project_iteration(storage_dir: str) -> int:
    return CacheContext.query_cache(storage_dir, 'num_iteration')


class CacheContext:
    def __init__(self, config: Config, storage_dir: str):
        self.config = config
        self.storage_dir = storage_dir
        self.num_iteration = None

    def __enter__(self):
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        self.write_to_cache(self.storage_dir, 'initialized', True, check_context=False)

        self.num_iteration = self.query_cache(self.storage_dir, 'num_iteration')
        if self.num_iteration is None:
            self.num_iteration = 0
            self.write_to_cache(self.storage_dir, 'num_iteration', self.num_iteration)

        # Clear possible residuals from last runs.
        cache_experiment_tracker_name(self.storage_dir, '')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Increment num_iteration for the next run.
        self.write_to_cache(self.storage_dir, 'num_iteration', self.num_iteration + 1)
        # Clear any possible residuals.
        cache_experiment_tracker_name(self.storage_dir, '')
        # After initialized is set to 'False' no more operations should be done on the cache.
        self.write_to_cache(self.storage_dir, 'initialized', False)

    @classmethod
    def write_to_cache(
            cls,
            storage_dir: str,
            key: str,
            value: Any,
            local_cache: dict = None,
            check_context: bool = True
    ):
        cache_file_path = build_cache_path(storage_dir)

        if local_cache is None:
            local_cache = cls.get_local_cache(storage_dir, check_context)

        local_cache[key] = value
        with open(cache_file_path, 'w') as f:
            json.dump(local_cache, f)

    @classmethod
    def query_cache(cls, storage_dir: str, key: str) -> Any:
        local_cache = cls.get_local_cache(storage_dir)

        return local_cache.get(key, None)

    @classmethod
    def get_local_cache(cls, storage_dir: str, check_context: bool = True) -> dict:
        cache_file_path = build_cache_path(storage_dir)

        if check_context and not os.path.exists(cache_file_path):
            raise RuntimeError(f'{cls.__class__.__name__} is not initialized.')
        elif not os.path.exists(cache_file_path):
            return dict()

        with open(cache_file_path, 'r') as f:
            local_cache = json.load(f)

        if check_context and local_cache.get('initialized', False) is False:
            raise RuntimeError(f'{cls.__class__.__name__} is not initialized.')

        return local_cache
