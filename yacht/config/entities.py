import datetime


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
    def seconds(self) -> int:
        """
        Returns:
            Data frequency in seconds.
        """
        type_to_seconds_mapping = {
            'S': 1,
            'M': 60,
            'H': 60 * 60,
            'D': 24 * 60 * 60
        }
        assert self.supported_types.keys() == type_to_seconds_mapping.keys()

        return self.value * type_to_seconds_mapping[self.type]

    @property
    def timedelta(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.seconds)
