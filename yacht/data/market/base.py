from datetime import datetime


class BaseMarket:
    def get(self, start: datetime, end: datetime, ticker: str):
        pass
