from abc import ABC, abstractmethod
from typing import Any, List, Optional

import pandas as pd

from yacht.config import Config


class Transform(ABC):
    @abstractmethod
    def __call__(self, sample: Any) -> Any:
        pass


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, sample: Any) -> Any:
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class RelativeClosePriceScaling:
    PRICE_COLUMNS = ['Close', 'Open', 'High', 'Low']

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.PRICE_COLUMNS] = data[self.PRICE_COLUMNS] / data['Close'].iloc[-1]
        data['Volume'] = data['Volume'] / data['Volume'].iloc[-1]

        return data


class AverageValueDiff:
    COLUMNS = ['Close', 'Open', 'High', 'Low', 'Volume']

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        columns_average = data[self.COLUMNS].mean()
        data[self.COLUMNS] = data[self.COLUMNS] - columns_average

        return data

#######################################################################################################################


transforms_registry = {
    'RelativeClosePriceScaling': RelativeClosePriceScaling,
    'AverageValueDiff': AverageValueDiff
}


def build_transforms(config: Config) -> Optional[Compose]:
    input_config = config.input
    if len(input_config.window_transforms) == 0:
        return None

    transforms = [transforms_registry[name]() for name in input_config.window_transforms]

    return Compose(transforms=transforms)
