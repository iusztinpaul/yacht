from dataclasses import dataclass
from datetime import datetime

import yaml

import utils
from entities import Frequency


class Config:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as file:
            configuration = yaml.full_load(file)

        self.input_config = InputConfig(
            market=configuration['input']['market'],
            start_datetime=utils.str_to_datetime(configuration['input']['start_datetime']),
            end_datetime=utils.str_to_datetime(configuration['input']['end_datetime']),
            data_frequency=Frequency(configuration['input']['data_frequency']),
            window_size=configuration['input']['window_size']
        )
        self.training_config = TrainingConfig(
            steps=configuration['training']['steps'],
            learning_rate=configuration['training']['learning_rate'],
            batch_size=configuration['training']['batch_size'],
            buffer_biased=configuration['training']['buffer_biased'],
            optimizer=configuration['training']['optimizer'],
            loss_function=configuration['training']['loss_function']
        )


@dataclass
class InputConfig:
    market: str
    start_datetime: datetime
    end_datetime: datetime
    data_frequency: Frequency
    window_size: int

    @property
    def start_datetime_seconds(self) -> int:
        return utils.datetime_to_seconds(self.start_datetime)

    @property
    def end_datetime_seconds(self) -> int:
        return utils.datetime_to_seconds(self.end_datetime)

    @property
    def data_span(self) -> int:
        return int((self.end_datetime_seconds - self.start_datetime_seconds) / self.data_frequency.seconds) + 1


@dataclass
class TrainingConfig:
    steps: int
    learning_rate: float
    batch_size: int
    buffer_biased: float
    optimizer: str
    loss_function: str
