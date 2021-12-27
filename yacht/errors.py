from typing import Optional


class DownloadError(RuntimeError):
    def __init__(self, ticker: Optional[str] = None):
        if ticker is not None:
            message = f'Could not download {ticker}'
        else:
            message = 'Data could not be downloaded'
        super().__init__(message)


class PreProcessError(RuntimeError):
    def __init__(self, ticker: Optional[str] = None):
        if ticker is not None:
            message = f'Could not pre-process {ticker}'
        else:
            message = f'Data could not be pre-processed'
        super().__init__(message)
