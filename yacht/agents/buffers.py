from typing import Optional, NamedTuple, Generator

import numpy as np
import torch as th

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize


class StudentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    teacher_actions: th.Tensor
    action_probabilities: th.Tensor


class StudentRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher_actions = None
        self.action_probabilities = None

    def reset(self) -> None:
        # (buffer_size, n_envs, num_assets)
        self.teacher_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.int_)
        # (buffer_size, n_envs, num_action_possibilities, num_assets)
        self.action_probabilities = np.zeros(
            (self.buffer_size, self.n_envs, *self.action_space.nvec, self.action_dim),
            dtype=np.float32
        )

        super().reset()

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
            teacher_action: np.ndarray,
            action_probabilities: th.Tensor
    ) -> None:
        self.teacher_actions[self.pos] = np.array(teacher_action, dtype=np.int_).copy()
        self.action_probabilities[self.pos] = action_probabilities.clone().cpu().numpy()

        super().add(
            obs,
            action,
            reward,
            episode_start,
            value,
            log_prob
        )

    def get(self, batch_size: Optional[int] = None) -> Generator[StudentRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "teacher_actions",
                "action_probabilities"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> StudentRolloutBufferSamples:
        sample = super()._get_samples(batch_inds, env)
        additional_sample = (
            self.teacher_actions[batch_inds],
            self.action_probabilities[batch_inds]
        )
        additional_sample = tuple(map(self.to_torch, additional_sample))
        sample = sample + additional_sample

        return StudentRolloutBufferSamples(*sample)
