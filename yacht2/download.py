import argparse
import logging
from pathlib import Path

from tqdm import tqdm

import utils
from data.market import build_market
from yacht2.config import Config

logger = logging.getLogger(__file__)


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True, help='Path to your *.yaml configuration file.')
parser.add_argument("--storage_path", required=True, help='Path to the directory where your model & logs will be saved.')
parser.add_argument("--logger_level", default='info', choices=('info', 'debug', 'warn'))


logger_levels = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN
}


if __name__ == '__main__':
    args = parser.parse_args()
    Path(args.storage_path).mkdir(parents=True, exist_ok=True)
    utils.setup_logger(level=args.logger_level)

    config = Config(args.config_file)

    logging.info(f'Starting downloading data from: '
                 f'{config.input_config.start_datetime} --> {config.input_config.end_datetime}')

    market = build_market(config=config)
    download_span = utils.compute_datetime_span(
        config.input_config.start_datetime,
        config.input_config.end_datetime,
        market.max_download_timedelta
    )

    with tqdm(total=len(download_span)) as prog_bar:
        for start, end in zip(download_span[:-1], download_span[1:]):
            market.download(
                start=start,
                end=end
            )
            prog_bar.update()
    logging.info('Finished downloading')
