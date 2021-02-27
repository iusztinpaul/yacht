from typing import Tuple, Union

import numpy as np

from config import Config
from data.loaders import BaseDataLoader
from data.market import BaseMarket


class XYDataLoader(BaseDataLoader):
    def __init__(
            self,
            market: BaseMarket,
            config: Config,
    ):
        super().__init__(market=market, config=config, window_size_offset=1)

    def next_batch(self) -> Union[np.array, Tuple[np.array]]:
        batch = super().next_batch()

        X = batch[:, :, :, :-1]
        y = batch[:, :, :, -1] / batch[:, 0, None, :, -2]

        return X, y






