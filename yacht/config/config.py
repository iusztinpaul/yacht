from dataclasses import dataclass
from datetime import datetime
from typing import List

import yaml

import utils
from entities import Frequency


class Config:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as file:
            configuration = yaml.full_load(file)

        self.hardware_config = HardwareConfig(
            device=configuration['hardware']['device']
        )
        self.input_config = InputConfig(
            market=configuration['input']['market'],
            start_datetime=utils.str_to_datetime(configuration['input']['start_datetime']),
            end_datetime=utils.str_to_datetime(configuration['input']['end_datetime']),
            data_frequency=Frequency(configuration['input']['data_frequency']),
            window_size=configuration['input']['window_size'],
            validation_split=float(configuration['input']['validation_split']),
            render_prices=bool(configuration['input']['render_prices'])
        )
        self.training_config = TrainingConfig(
            agent=configuration['training']['agent'],
            steps=configuration['training']['steps'],
            validation_every_step=configuration['training']['validation_every_step'],
            save_every_step=configuration['training']['save_every_step'],
            learning_rate=float(configuration['training']['learning_rate']),
            weight_decay=float(configuration['training']['weight_decay']),
            learning_rate_decay=float(configuration['training']['learning_rate_decay']),
            learning_rate_decay_steps=configuration['training']['learning_rate_decay_steps'],
            batch_size=configuration['training']['batch_size'],
            buffer_biased=float(configuration['training']['buffer_biased']),
            optimizer=configuration['training']['optimizer'],
            loss_function=configuration['training']['loss_function']
        )


@dataclass
class HardwareConfig:
    device: str


@dataclass
class InputConfig:
    market: str
    start_datetime: datetime
    end_datetime: datetime
    data_frequency: Frequency
    window_size: int
    validation_split: float
    render_prices: bool

    @property
    def start_datetime_seconds(self) -> int:
        return utils.datetime_to_seconds(self.start_datetime)

    @property
    def end_datetime_seconds(self) -> int:
        return utils.datetime_to_seconds(self.end_datetime)

    @property
    def data_span(self) -> List[int]:
        return list(range(
            self.start_datetime_seconds,
            self.end_datetime_seconds + 1,
            self.data_frequency.seconds
        ))


@dataclass
class TrainingConfig:
    agent: str
    steps: int
    validation_every_step: int
    save_every_step: int
    learning_rate: float
    weight_decay: float
    learning_rate_decay: float
    learning_rate_decay_steps: int
    batch_size: int
    buffer_biased: float
    optimizer: str
    loss_function: str
