from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from gym import Space

from yacht.data.datasets import AssetDataset, SingleAssetDataset
from yacht.data.markets import Market


class ChooseAssetDataset(AssetDataset):
    def __init__(
            self,
            datasets: List[SingleAssetDataset],
            market: Market,
            intervals: List[str],
            features: List[str],
            start: datetime,
            end: datetime,
            window_size: int = 1,
            default_ticker: str = None,
    ):
        super().__init__(
            market=market,
            intervals=intervals,
            features=features,
            start=start,
            end=end,
            window_size=window_size,
        )

        self.datasets = datasets
        self.tickers = [dataset.ticker for dataset in self.datasets]
        self.default_ticker = default_ticker
        self.current_ticker, self.current_ticker_index = self.choose_ticker(default_ticker)

    def choose_ticker(self, ticker: Optional[str] = None) -> Tuple[str, int]:
        if ticker is None:
            ticker = np.random.choice(self.tickers)

        idx = self.tickers.index(ticker)
        assert idx != -1, f'Ticker not supported: {ticker}'

        self.current_ticker = ticker
        self.current_ticker_index = idx

        return ticker, idx

    @property
    def num_days(self) -> int:
        return self.datasets[self.current_ticker_index].num_days

    def index_to_datetime(self, integer_index: int) -> datetime:
        return self.datasets[self.current_ticker_index].index_to_datetime(integer_index)

    def get_prices(self) -> pd.DataFrame:
        return self.datasets[self.current_ticker_index].get_prices()

    def get_external_observation_space(self) -> Dict[str, Space]:
        return self.datasets[self.current_ticker_index].get_external_observation_space()

    def __len__(self):
        return len(self.datasets[self.current_ticker_index])

    def __getitem__(self, item):
        return self.datasets[self.current_ticker_index][item]
