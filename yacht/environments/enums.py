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


class Mode(Enum):
    Train = 'train'
    Validation = 'validation'
    TrainVal = 'trainval'
    Backtest = 'backtest'
    BacktestTrain = 'backtest_train'
    Baseline = 'baseline'

    @classmethod
    def from_string(cls, value: str) -> 'Mode':
        return cls(value.lower())

    def is_trainval(self) -> bool:
        return self in (self.Train, self.Validation, self.TrainVal)
