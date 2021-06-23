from enum import Enum


class Position(Enum):
    Short = -1
    Hold = 0
    Long = 1

    @classmethod
    def build(cls, position: int):
        if position > 0:
            return Position.Long
        elif position == 0:
            return Position.Hold
        else:
            return Position.Short

    def opposite(self):
        return Position.Short if self == Position.Long else Position.Long
