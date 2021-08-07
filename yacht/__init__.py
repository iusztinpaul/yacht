from enum import Enum


class Mode(Enum):
    Train = 'train'
    BacktestTrain = 'backtest_on_train'
    BacktestValidation = 'validation_on_validation'
    Backtest = 'backtest_on_test'


    @classmethod
    def from_string(cls, value: str) -> 'Mode':
        if value == 'backtest':
            value = 'backtest_on_test'

        return cls(value.lower())

    def is_trainable(self) -> bool:
        return self == self.Train

    def is_backtest_on_train(self) -> bool:
        return self == self.BacktestTrain

    def is_validation(self) -> bool:
        return self == self.BacktestValidation

    def is_trainval(self) -> bool:
        return any([self.is_trainable(), self.is_validation()])
