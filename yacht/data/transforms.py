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


class RelativeNormalization:
    PRICE_COLUMNS = ['Close', 'Open', 'High', 'Low']

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        other_columns = list(set(data.columns) - set(self.PRICE_COLUMNS))
        data[self.PRICE_COLUMNS] = data[self.PRICE_COLUMNS].values / data['Close'].iloc[-1]
        if len(other_columns) > 0:
            data[other_columns] = data[other_columns].values / data[other_columns].values[-1, :]

        return data


#######################################################################################################################


transforms_registry = {
    'RelativeNormalization': RelativeNormalization
}


def build_transforms(config: Config) -> Optional[Compose]:
    input_config = config.input
    if len(input_config.window_transforms) == 0:
        return None

    transforms = [transforms_registry[name]() for name in input_config.window_transforms]

    return Compose(transforms=transforms)
