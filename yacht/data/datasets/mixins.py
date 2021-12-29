from typing import Dict

import numpy as np
import pandas as pd
from gym import spaces


class MetaLabelingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.labels = self.compute_labels()

    def compute_labels(self) -> np.ndarray:
        # TODO: Adapt for sell execution.
        mean_price = self.compute_mean_price(
            start=self.start,
            end=self.end
        ).item()
        decision_prices = self.get_decision_prices()

        labels = decision_prices < mean_price
        labels = labels.astype(np.int32)

        return labels


class AttachDataMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attached_tickers = ['QQQ']
        for attached_ticker in self.attached_tickers:
            for interval in self.intervals:
                if self.market.is_cached(attached_ticker, interval, self.start, self.end):
                    attached_ticker_data = self.market.get(
                        ticker=attached_ticker,
                        interval=interval,
                        start=self.start,
                        end=self.end,
                        features=self.features,
                        squeeze=False
                    )
                    new_columns = {
                        column: f'{column}{attached_ticker}{interval}' for column in attached_ticker_data.columns
                    }
                    attached_ticker_data = attached_ticker_data.rename(columns=new_columns)
                    self.data[interval] = pd.concat([
                        self.data[interval],
                        attached_ticker_data
                    ], axis=1)

    def get_external_observation_space(self) -> Dict[str, spaces.Space]:
        observation_space = super().get_external_observation_space()

        for item in observation_space.values():
            old_shape = item.shape
            item.shape = (*old_shape[:-1], old_shape[-1] * (len(self.attached_tickers) + 1))

        return observation_space
