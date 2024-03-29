from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from gym import spaces
from pandas import Interval

from yacht import Mode
from yacht.data.datasets import SingleAssetDataset, DatasetPeriod
from yacht.data.datasets.mixins import MetaLabelingMixin
from yacht.data.markets import Market
from yacht.data.scalers import Scaler
from yacht.data.transforms import Compose
from yacht.logger import Logger


class DayFrequencyDataset(MetaLabelingMixin, SingleAssetDataset):
    def __init__(
            self,
            ticker: str,
            market: Market,
            storage_dir: str,
            intervals: List[str],
            features: List[str],
            decision_price_feature: str,
            period: DatasetPeriod,
            render_intervals: List[Interval],
            render_tickers: List[str],
            mode: Mode,
            logger: Logger,
            scaler: Scaler,
            window_transforms: Optional[Compose] = None,
            window_size: int = 1,
            data: Dict[str, pd.DataFrame] = None
    ):
        assert set(intervals) == {'1d'}, 'Requested intervals are not supported.'

        super().__init__(
            ticker=ticker,
            market=market,
            storage_dir=storage_dir,
            intervals=intervals,
            features=features,
            decision_price_feature=decision_price_feature,
            period=period,
            render_intervals=render_intervals,
            render_tickers=render_tickers,
            mode=mode,
            logger=logger,
            scaler=scaler,
            window_transforms=window_transforms,
            window_size=window_size,
            data=data
        )

    def __len__(self):
        return len(self.data['1d'])

    def get_external_observation_space(self) -> Dict[str, spaces.Space]:
        return {
            '1d': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.window_size, 1, len(self.features)),  # (window, bar, features)
                dtype=np.float32
            )
        }

    def __getitem__(self, day_index: int) -> Dict[str, np.array]:
        """
        Args:
            day_index: The relative index the data will be given from.

        Returns:
            The data features within the [day_index - window_size + 1, day_index] interval.
        """

        day_features = self.data['1d'][self.features]
        start_index = day_index - self.window_size + 1
        end_index = day_index + 1

        day_features = day_features.iloc[start_index:end_index]
        if self.window_transforms is not None:
            day_features = self.window_transforms(day_features)
        day_features = self.scaler.transform(day_features)
        day_features = np.expand_dims(day_features, axis=1)

        return {
            '1d': day_features
        }
