from datetime import datetime

from .base import BaseDataLoader


class BackTestDataLoader(BaseDataLoader):
    def get_batch_size(self) -> int:
        return self.input_config.batch_size

    def get_first_batch_start_datetime(self) -> datetime:
        return self.input_config.start_datetime
