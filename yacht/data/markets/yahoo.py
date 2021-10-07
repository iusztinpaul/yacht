from datetime import datetime
from typing import List, Any, Union

import pandas as pd
import yfinance

from yacht.data.markets.base import H5Market
from yacht.data.markets.mixins import TechnicalIndicatorMixin, TargetPriceMixin
from yacht.logger import Logger


class Yahoo(H5Market):
    def __init__(
            self,
            features: List[str],
            logger: Logger,
            api_key,
            api_secret,
            storage_dir: str,
            include_weekends: bool,
            read_only: bool
    ):
        super().__init__(features, logger, api_key, api_secret, storage_dir, 'yahoo.h5', include_weekends, read_only)

    def request(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime = None
    ) -> Union[List[List[Any]], pd.DataFrame]:
        assert interval == '1d', 'Yahoo Finance supports only interval = "1d".'

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


class TechnicalIndicatorYahoo(TechnicalIndicatorMixin, TargetPriceMixin, Yahoo):
    pass
