import glob
import os
import logging
from functools import partial
from pathlib import Path

import torch

from typing import Tuple

from torch import optim

from config import Config
from environment.environment import Environment


logger = logging.getLogger(__file__)


class BaseAgent:
    def __init__(
            self,
            environment: Environment,
            config: Config,
            storage_path: str,
    ):
        self.environment = environment
        self.config = config

        self.storage_path = storage_path
        self.model_path = os.path.join(self.storage_path, 'model')

        self.strategy = self.build_strategy(environment, config)

        self.current_step = 0

    def build_strategy(self, environment: Environment, config: Config):
        raise NotImplementedError()

    def _call_loader(self, loader_function):
        hardware_config = self.config.hardware_config

        X, y, last_w, batch_new_w_datetime = loader_function()

        X = torch.from_numpy(X).to(hardware_config.device)
        y = torch.from_numpy(y).to(hardware_config.device)
        last_w = torch.from_numpy(last_w).to(hardware_config.device)

        return X, y, last_w, batch_new_w_datetime

    def load_network(self):
        def _extract_step_from(checkpoint_path: str) -> int:
            return int(os.path.split(checkpoint_path)[1].split('.')[0].split('_')[1])

        checkpoints = sorted(
            glob.glob(os.path.join(self.model_path, '*.checkpoint')),
            key=lambda checkpoint_path: _extract_step_from(checkpoint_path)
        )
        if len(checkpoints) > 0:
            last_checkpoint_path = checkpoints[-1]

            checkpoint = torch.load(last_checkpoint_path)

            logger.info(f'Resuming training from step {checkpoint["step"]} out of {self.config.training_config.steps}')
            self.strategy.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            return checkpoint['step']
        else:
            raise RuntimeError(f'No checkpoint to load from {self.model_path}. Maybe set "resume_training" to "false"')

    def log(self, step, message: str):
        total_steps = self.config.training_config.steps

        logger.info(f'Step [{step} - {round(step / total_steps, 4) * 100}%] - {message}')


class TrainBaseAgent(BaseAgent):
    def __init__(
            self,
            environment: Environment,
            config: Config,
            storage_path: str,
            resume_training: bool = False
    ):
        super(TrainBaseAgent, self).__init__(environment, config, storage_path)

        self.optimizer, self.scheduler = self.build_scheduler()
        self.loss = None

        self.start_training_step = 0
        self.resume_training = resume_training
        if resume_training:
            self.start_training_step = self.load_network()

    def train(self):
        raise NotImplementedError()

    def get_train_data(self):
        return self._call_loader(partial(self.environment.next, reason='train'))

    def get_validation_data(self):
        return self._call_loader(partial(self.environment.next, reason='validation'))

    def build_scheduler(self) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        training_config = self.config.training_config

        optimizer = optim.Adam(
            params=self.strategy.params,
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )

        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=training_config.learning_rate_decay,
        )

        return optimizer, scheduler

    def save_network(self, step: int, loss: torch.Tensor):
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        checkpoint_path = os.path.join(self.model_path, f'step_{step}.checkpoint')

        torch.save(
            {
                'step': step,
                'model_state_dict': self.strategy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            },
            checkpoint_path
        )


class BackTestBaseAgent(BaseAgent):
    def trade(self):
        raise NotImplementedError()

    def get_back_test_data(self):
        return self._call_loader(partial(self.environment.next, reason='back_test'))
