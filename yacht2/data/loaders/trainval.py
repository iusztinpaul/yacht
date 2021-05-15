from datetime import datetime

from config import InputConfig, TrainingConfig
from data.loaders import BaseDataLoader
from data.market import BaseMarket
from data.memory_replay import MemoryReplayBuffer
from data.renderers import BaseRenderer


class TrainDataLoader(BaseDataLoader):
    def __init__(
            self,
            market: BaseMarket,
            renderer: BaseRenderer,
            input_config: InputConfig,
            window_size_offset: int = 1,
    ):
        super().__init__(
            market=market,
            renderer=renderer,
            input_config=input_config,
            window_size_offset=window_size_offset,
        )

        self.memory_replay = MemoryReplayBuffer.from_config(input_config, window_size_offset)

    def get_batch_size(self) -> int:
        return self.input_config.batch_size

    def get_first_batch_start_datetime(self) -> datetime:
        return self.memory_replay.get_experience()


class ValidationDataLoader(BaseDataLoader):
    def get_batch_size(self) -> int:
        return len(self.input_config.data_span) - self.input_config.window_size - 1

    def get_first_batch_start_datetime(self) -> datetime:
        return self.input_config.start_datetime
