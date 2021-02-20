import time
from datetime import datetime

import yaml


class Config:
    def __init__(self, file_path: str):
        with open(file_path, 'r') as file:
            self.configuration = yaml.full_load(file)

    @property
    def data_time_span(self):
        start_timestamp = self.parse_time(self.configuration['input']['start_date'])
        end_timestamp = self.parse_time(self.configuration['input']['end_date'])
        data_frequency = self.parse_data_frequency()

        return (end_timestamp - start_timestamp) // data_frequency

    def parse_time(self, t: str):
        return time.mktime(datetime.strptime(t, "%Y/%m/%d").timetuple())

    def parse_data_frequency(self):
        """
        Returns:
            The data frequency in seconds
        """
        time_type = self.configuration['input']['data_frequency'][-1]
        time_period = int(self.configuration['input']['data_frequency'][:-1])

        time_to_seconds = {
            'S': 1,
            'M': 60,
            'H': 60 * 60,
            'D': 24 * 60 * 60
        }[time_type]

        return time_period * time_to_seconds
