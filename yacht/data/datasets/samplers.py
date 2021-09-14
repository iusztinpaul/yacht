import random
from datetime import datetime
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
from gym import Space
from pandas import Interval

from yacht import Mode
from yacht.data.datasets import AssetDataset, SingleAssetDataset, MultiAssetDataset
from yacht.data.markets import Market
from yacht.logger import Logger


class SampleAssetDataset(AssetDataset):
    def __init__(
            self,
            datasets: List[Union[SingleAssetDataset, MultiAssetDataset]],
            market: Market,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            start: datetime,
            end: datetime,
            render_intervals: List[Interval],
            mode: Mode,
            logger: Logger,
            window_size: int = 1,
            default_index: int = 0,
            shuffle: bool = False
    ):
        super().__init__(
            market=market,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            start=start,
            end=end,
            render_intervals=render_intervals,
            mode=mode,
            logger=logger,
            window_size=window_size,
        )

        self.datasets = datasets
        self.shuffle = shuffle

        # Keep a map of the dataset sampling order.
        self.dataset_indices = np.arange(0, len(self.datasets))
        self.current_dataset_indices_position = 0
        if self.shuffle:
            self.seed(self.default_index)
            np.random.shuffle(self.dataset_indices)

        self.default_index = default_index
        self.current_dataset_index = self.sample(default_index)
        
    def seed(self, value: float):
        # Seed python RNG
        random.seed(value)
        # Seed numpy RNG
        np.random.seed(value)

    def sample(self, idx: Optional[int] = None, seed: Optional[float] = None) -> int:
        if seed is not None:
            self.seed(seed)
            
        if idx is None:
            # Move to the next dataset by moving the map cursor.
            self.current_dataset_indices_position += 1
            if self.current_dataset_indices_position == len(self.datasets):
                self.current_dataset_indices_position = 0
                if self.shuffle:
                    np.random.shuffle(self.dataset_indices)

            # Get the actual dataset index from the map.
            idx = self.dataset_indices[self.current_dataset_indices_position]

        self.current_dataset_index = idx

        return idx

    @property
    def num_sampling_period(self) -> int:
        return (self.end - self.start).days

    @property
    def sampled_dataset(self) -> Union[SingleAssetDataset, MultiAssetDataset]:
        # Expose the chosen dataset to access its attributes.
        return self.datasets[self.current_dataset_index]

    @property
    def should_render(self) -> bool:
        return self.datasets[self.current_dataset_index].should_render

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

    def get_decision_prices(self, t_tick: Optional[int] = None, ticker: Optional[str] = None) -> pd.Series:
        return self.datasets[self.current_dataset_index].get_decision_prices(t_tick, ticker)

    def compute_mean_price(self, start: datetime, end: datetime) -> Union[pd.DataFrame, pd.Series]:
        return self.datasets[self.current_dataset_index].compute_mean_price(start, end)

    def get_external_observation_space(self) -> Dict[str, Space]:
        return self.datasets[self.current_dataset_index].get_external_observation_space()

    def inverse_scaling(self, observation: dict, **kwargs) -> dict:
        return self.datasets[self.current_dataset_index].inverse_scaling(observation, **kwargs)

    def __len__(self):
        return len(self.datasets[self.current_dataset_index])

    def __getitem__(self, item):
        return self.datasets[self.current_dataset_index][item]

    def __str__(self) -> str:
        return self.datasets[self.current_dataset_index].__str__()
