from datetime import datetime
from typing import List, Any, Union

import pandas as pd
import yfinance

from yacht.data.markets.base import H5Market
from yacht.data.markets.mixins import TechnicalIndicatorMixin


class Yahoo(H5Market):
    def __init__(
            self,
            features: List[str],
            api_key,
            api_secret,
            storage_dir: str
    ):
        super().__init__(features, api_key, api_secret, storage_dir, 'yahoo.h5')

    def request(self, ticker: str, interval: str, start: datetime, end: datetime = None) -> List[List[Any]]:
        ticker_data = yfinance.download(
            tickers=[ticker],
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            prepost=False
        )

        return ticker_data

    def process_request(self, data: Union[List[List[Any]], pd.DataFrame]) -> pd.DataFrame:
        data = data.loc[:, self.MANDATORY_FEATURES]

        return data


class TechnicalIndicatorYahoo(TechnicalIndicatorMixin, Yahoo):
    pass
