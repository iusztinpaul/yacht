from enum import Enum


class Mode(Enum):
    Train = 'train'
    Validation = 'validation'
    Backtest = 'backtest_on_test'
    BacktestTrain = 'backtest_on_train'
    Baseline = 'baseline'

    @classmethod
    def from_string(cls, value: str) -> 'Mode':
        return cls(value.lower())

    def is_trainable(self) -> bool:
        return self == self.Train

    def is_backtest_on_train(self) -> bool:
        return self == self.BacktestTrain

    def is_validation(self) -> bool:
        return self == self.Validation

    def is_trainval(self) -> bool:
        return any([self.is_trainable(), self.is_validation()])
