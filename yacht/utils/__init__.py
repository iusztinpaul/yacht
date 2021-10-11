from .paths import *
from .cache import *
from .misc import *
from .parsers import *

from pathlib import Path

from dotenv import load_dotenv

from yacht.config import Config


def load_env_variables(root_dir: str):
    env_path = Path(root_dir) / '.env.default'
    load_dotenv(dotenv_path=env_path)

    env_path = Path(root_dir) / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)


def create_project_name(config: Config, storage_dir: str):
    project_iteration = get_project_iteration(storage_dir)
    name = f'{os.path.basename(storage_dir)}_{project_iteration}'

    return name
