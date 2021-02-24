import time
from dataclasses import dataclass
from datetime import datetime

import yaml

import utils


class Frequency:
    supported_types = {
        'S': 'second',
        'M': 'minute',
        'H': 'hour',
        'D': 'day'
    }

    def __init__(self, value_type_frequency: str):
        """
        Args:
            value_type_frequency: A string of the form "V..VT", where 'V' is the value of the frequency &
                'T' is the time measurement: e.g. 1D, 600S, 15H etc.
        """
        self.type = value_type_frequency[-1]
        if self.type not in self.supported_types.keys():
            raise RuntimeError(f'Wrong frequency type: {self.type}.')

        self.value = int(value_type_frequency[:-1])

    @property
    def seconds(self):
        """
        Returns:
            Data frequency in seconds.
        """
        time_to_seconds_mapping = {
            'S': 1,
            'M': 60,
            'H': 60 * 60,
            'D': 24 * 60 * 60
        }
        assert self.supported_types.keys() == time_to_seconds_mapping.keys()

        return self.value * time_to_seconds_mapping[self.type]


class Config:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as file:
            self.configuration = yaml.full_load(file)

        self.input_config = InputConfig(
            window_size=self.configuration['input']['window_size'],
            start_date=utils.str_to_datetime(self.configuration['input']['start_date']),
            end_date=utils.str_to_datetime(self.configuration['input']['end_date']),
            data_frequency=Frequency(self.configuration['input']['data_frequency']),
        )


@dataclass
class InputConfig:
    window_size: int
    start_date: datetime
    end_date: datetime
    data_frequency: Frequency

    @property
    def start_date_seconds(self) -> int:
        return utils.datetime_to_seconds(self.start_date)

    @property
    def end_date_seconds(self) -> int:
        return utils.datetime_to_seconds(self.end_date)

    @property
    def data_span_seconds(self) -> int:
        return int((self.end_date_seconds - self.start_date_seconds) / self.data_frequency.seconds)
