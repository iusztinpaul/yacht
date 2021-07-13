import json
import os
from typing import Any

from yacht.utils import build_cache_path


def cache_experiment_tracker_name(storage_dir: str, experiment_tracker_name: str):
    write_to_cache(storage_dir, 'experiment_tracker_name', experiment_tracker_name)


def get_experiment_tracker_name(storage_dir: str) -> str:
    return query_cache(storage_dir, 'experiment_tracker_name')


def write_to_cache(storage_dir: str, key: str, value: str, local_cache: dict = None):
    cache_file_path = build_cache_path(storage_dir)

    if local_cache is None:
        local_cache = get_local_cache(storage_dir)

    local_cache[key] = value
    with open(cache_file_path, 'w') as f:
        json.dump(local_cache, f)


def get_local_cache(storage_dir: str) -> dict:
    cache_file_path = build_cache_path(storage_dir)

    if not os.path.exists(cache_file_path):
        return dict()

    with open(cache_file_path, 'r') as f:
        local_cache = json.load(f)

    return local_cache


def query_cache(storage_dir: str, key: str) -> Any:
    local_cache = get_local_cache(storage_dir)

    return local_cache.get(key, None)
