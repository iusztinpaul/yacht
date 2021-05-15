from google.protobuf import text_format

from yacht.config.proto import *


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config_str = f.read()
        config = Config()
        text_format.Merge(config_str, config)

    return config
