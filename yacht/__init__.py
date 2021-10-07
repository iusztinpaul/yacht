from enum import Enum


__author__ = "Iusztin Paul-Emil"
__copyright__ = "Copyright 2007, Yacht"
__credits__ = ["Iusztin Paul-Emil"]

__license__ = "Apache-2.0"
__version__ = "1.0.0"
__maintainer__ = "Iusztin Paul-Emil"
__email__ = "p.e.iusztin@gmail.com"
__status__ = "Development"


class Mode(Enum):
    Download = 'download'
    Train = 'train'
    FineTuneTrain = 'fine_tune_train'
    BacktestTrain = 'backtest_on_train'
    BacktestValidation = 'backtest_on_validation'
    Backtest = 'backtest_on_test'

    @classmethod
    def from_string(cls, value: str) -> 'Mode':
        if value == 'backtest':
            value = 'backtest_on_test'

        return cls(value.lower())

    def is_trainable(self) -> bool:
        return self == self.Train or self == self.FineTuneTrain

    def is_fine_tuning(self) -> bool:
        return self == self.FineTuneTrain

    def is_backtest_on_train(self) -> bool:
        return self == self.BacktestTrain

    def is_validation(self) -> bool:
        return self == self.BacktestValidation

    def is_backtestable(self) -> bool:
        return self in {self.BacktestTrain, self.BacktestValidation, self.Backtest}

    def is_trainval(self) -> bool:
        return any([self.is_trainable(), self.is_validation()])

    def to_step_key(self) -> str:
        return f'{self.value}_step'
