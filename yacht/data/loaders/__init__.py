from copy import copy

import utils
from config import Config
from .base import *
from .trainval import TrainDataLoader, ValidationDataLoader


def build_data_loaders(market: BaseMarket, config: Config):
    train_input_config = copy(config.input_config)
    val_input_config = copy(config.input_config)

    (start_split_train, end_split_train), (start_split_val, end_split_val) = utils.split_datetime_span(
        start_datetime=config.input_config.start_datetime,
        end_datetime=config.input_config.end_datetime,
        split=config.input_config.validation_split,
        frequency=config.input_config.data_frequency
    )

    end_split_train = end_split_train - (
            config.input_config.data_frequency.timedelta * (config.input_config.window_size + 1)
    )
    train_input_config.start_datetime = start_split_train
    train_input_config.end_datetime = end_split_train
    train_input_config.validation_split = 0

    val_input_config.start_datetime = start_split_val
    val_input_config.end_datetime = end_split_val
    val_input_config.validation_split = 1

    train_data_loader = TrainDataLoader(
        market=market,
        input_config=train_input_config,
        training_config=config.training_config
    )
    val_data_loader = ValidationDataLoader(
        market=market,
        input_config=val_input_config
    )

    return train_data_loader, val_data_loader
