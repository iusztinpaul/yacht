import os
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yfinance


def get_sp_500_index_tickers() -> List[str]:
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    return table[0]['Symbol'].tolist()


def get_dow_index_tickers() -> List[str]:
    table = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')

    return table[1]['Symbol'].tolist()


def get_nasdaq_index_tickers() -> List[str]:
    table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')

    return table[3]['Ticker'].tolist()


def get_russell_2000_index_tickers() -> List[str]:
    csv_filename = download_file(
        'https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund'
    )

    # Clean header & foot noise.
    with open(csv_filename, 'r') as f:
        data = f.readlines()
        data = data[9:-16]
    with open(csv_filename, 'w') as f:
        f.writelines(data)
    table = pd.read_csv(csv_filename)

    return table['Ticker'].tolist()


def download_file(url):
    local_filename = url.split('/')[-1]
    local_filename = os.path.join(results_dir, local_filename)

    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)

    return local_filename


def check_tickers_availability(tickers: List[str], source: str) -> List[str]:
    valid_tickers = []
    if source == 'yahoo':
        tickers_data = yfinance.download(
            tickers=tickers,
            period='max',
            interval='1d',
            auto_adjust=True,
            group_by='ticker',
            threads=True,
            prepost=False
        )
        for ticker in tickers:
            try:
                ticker_data = tickers_data[ticker]
            except KeyError:
                continue

            ticker_data = ticker_data.dropna()
            if len(ticker_data) > 0:
                # Add tickers in the following format so can be copy-pasted into python lists easily.
                valid_tickers.append(f'"{ticker}",\n')
    else:
        raise RuntimeError(f'Does not support source: {source}')

    return valid_tickers


indexes = {
    'sp-500': get_sp_500_index_tickers,
    'dow': get_dow_index_tickers,
    'nasdaq-100': get_nasdaq_index_tickers,
    'russell-2000': get_russell_2000_index_tickers
}
results_dir = 'crawl_indices_results'
source = 'yahoo'

if __name__ == '__main__':
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    for index_name, ticker_callback in indexes.items():
        index_tickers = ticker_callback()

        print(f'{index_name} - checking validity of tickers on {source}...')
        valid_index_tickers = check_tickers_availability(index_tickers, source=source)
        print(f'{index_name} - {len(valid_index_tickers)} / {len(index_tickers)}')

        index_file = os.path.join(results_dir, f'{index_name}.txt')
        with open(index_file, 'w') as f:
            f.writelines(valid_index_tickers)
