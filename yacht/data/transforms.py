from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

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
    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = data / data[-1, :, 0]

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
