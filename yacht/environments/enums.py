from enum import Enum
from typing import List, Union

import numpy as np

from yacht import utils


class Position(Enum):
    Short = -1
    Hold = 0
    Long = 1

    @classmethod
    def build(cls, position: Union[float, int, List[float], np.ndarray]) -> Union['Position', List['Position']]:
        def _map_to_position(value: float):
            if value > 0:
                return Position.Long
            elif value == 0:
                return Position.Hold
            else:
                return Position.Short

        if utils.is_number(position):
            return _map_to_position(position)
        else:
            return [_map_to_position(p) for p in position]

    def opposite(self):
        return Position.Short if self == Position.Long else Position.Long
