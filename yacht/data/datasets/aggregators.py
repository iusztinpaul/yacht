from datetime import datetime
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd
from gym import Space

from yacht import Mode
from yacht.data.datasets import AssetDataset, SingleAssetDataset, MultiAssetDataset
from yacht.data.markets import Market
from yacht.logger import Logger


class ChooseAssetDataset(AssetDataset):
    def __init__(
            self,
            datasets: List[Union[SingleAssetDataset, MultiAssetDataset]],
            market: Market,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime,
            mode: Mode,
            logger: Logger,
            window_size: int = 1,
            default_index: int = 0,
    ):
        super().__init__(
            market=market,
            intervals=intervals,
            features=features,
            start=start,
            end=end,
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        self.datasets = datasets
        self.default_index = default_index
        self.current_dataset_index = self.choose(default_index)

    def choose(self, idx: Optional[int] = None) -> int:
        if idx is None:
            idx = np.random.randint(0, len(self.datasets))

        self.current_dataset_index = idx

        return idx

    @property
    def num_days(self) -> int:
        return self.datasets[self.current_dataset_index].num_days

    @property
    def num_assets(self) -> int:
        return self.datasets[self.current_dataset_index].num_assets

    @property
    def asset_tickers(self) -> List[str]:
        return self.datasets[self.current_dataset_index].asset_tickers

    def index_to_datetime(self, integer_index: int) -> datetime:
        return self.datasets[self.current_dataset_index].index_to_datetime(integer_index)

    def get_prices(self) -> pd.DataFrame:
        return self.datasets[self.current_dataset_index].get_prices()

    def get_mean_over_period(self, start: datetime, end: datetime) -> Union[pd.DataFrame, pd.Series]:
        return self.datasets[self.current_dataset_index].get_mean_over_period(start, end)

    def get_external_observation_space(self) -> Dict[str, Space]:
        return self.datasets[self.current_dataset_index].get_external_observation_space()

    def __len__(self):
        return len(self.datasets[self.current_dataset_index])

    def __getitem__(self, item):
        return self.datasets[self.current_dataset_index][item]

    def inverse_scaling(self, observation: dict, **kwargs) -> dict:
        return self.datasets[self.current_dataset_index].inverse_scaling(observation, **kwargs)

    def __str__(self) -> str:
        return self.datasets[self.current_dataset_index].__str__()
