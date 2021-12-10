from datetime import datetime, timedelta
from typing import List, Any, Union

import pandas as pd
import yfinance

from yacht import utils
from yacht.data.markets.base import H5Market
from yacht.logger import Logger


class Yahoo(H5Market):
    def __init__(
            self,
            get_features: List[str],
            logger: Logger,
            api_key,
            api_secret,
            storage_dir: str,
            include_weekends: bool,
            read_only: bool
    ):
        super().__init__(
            get_features, logger, api_key, api_secret, storage_dir, 'yahoo.h5', include_weekends, read_only
        )

    def request(
            self,
            ticker: str,
            interval: str,
            start: datetime,
            end: datetime = None
    ) -> Union[List[List[Any]], pd.DataFrame]:
        """

        Args:
            ticker:
            interval:
            start:
            end:

        Returns:
            Prices data for [start, end] and interval: "interval"
        """

        assert interval in {'1d', '1h'}, f'Yahoo Market does not support interval: {interval}'

        # Add another day to the end because yahoo gives data for [start, end).
        end = utils.add_days(end, action='+', include_weekends=self.include_weekends, offset=1)
        ticker_data = yfinance.download(
            tickers=[ticker],
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            prepost=False
        )
        if len(ticker_data) == 0:
            raise RuntimeError(f'Data for {ticker} - {interval} in [{start} - {end}] is not valid')

        # For simplicity, make all indices naive.
        ticker_data.index = ticker_data.index.tz_localize(None)
        if interval == '1h' and self.include_weekends is False:
            # Normalize all data within the trading hours [9:00, 15:00].
            ticker_data.index = ticker_data.index.map(
                lambda t: t.replace(minute=0, second=0, microsecond=0, nanosecond=0)
            )

        return ticker_data

    def process_request(self, data: Union[List[List[Any]], pd.DataFrame]) -> pd.DataFrame:
        data = data.loc[:, self.DOWNLOAD_MANDATORY_FEATURES]

        return data
