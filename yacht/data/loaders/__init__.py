from typing import Dict

from config import Config
from .backtest import BackTestDataLoader
from .base import *
from .trainval import TrainDataLoader, ValidationDataLoader


def get_data_loader_builder(reason: str):
    return {
        'train': build_train_data_loaders,
        'back_test': build_back_test_data_loader
    }[reason]


def build_train_data_loaders(market: BaseMarket, renderer: BaseRenderer, config: Config) -> Dict[str, BaseDataLoader]:
    train_input_config, val_input_config = config.input_config.split_span()

    train_data_loader = TrainDataLoader(
        market=market,
        renderer=renderer,
        input_config=train_input_config,
    )
    val_data_loader = ValidationDataLoader(
        market=market,
        renderer=renderer,
        input_config=val_input_config,
    )

    return {
        'train': train_data_loader,
        'validation': val_data_loader
    }


def build_back_test_data_loader(market: BaseMarket, renderer: BaseRenderer, config: Config) -> Dict[str, BaseDataLoader]:
    _, back_test_input_config = config.input_config.split_span()

    back_test_loader = BackTestDataLoader(
        market=market,
        renderer=renderer,
        input_config=back_test_input_config,
    )

    return {
        'back_test': back_test_loader
    }
