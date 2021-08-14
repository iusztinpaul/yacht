import os

from google.protobuf import text_format
from google.protobuf.text_format import MessageToString

from yacht.config.proto import *


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config_str = f.read()
        config = Config()
        text_format.Merge(config_str, config)

    return config


def export_config(config: Config, storage_dir: str):
    if not os.path.exists(storage_dir):
        os.mkdir(storage_dir)

    with open(os.path.join(storage_dir, 'config.txt'), 'w') as f:
        config_txt = MessageToString(config)
        f.write(config_txt)
