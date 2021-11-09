import numpy as np
from stable_baselines3.common import noise
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise

from yacht.config import Config
from yacht.utils import build_from_protobuf


class ActionDropOut(ActionNoise):
    def __init__(self, p: float = 0.25):
        super().__init__()
        assert 0 <= p <= 1

        self.p = 1. - p
        self.operation = '*'

    def __call__(self):
        noise_value = np.random.binomial(1, p=self.p, size=1)

        return noise_value


# Action Noise Utilities ----------------------------------------------------------------------------------------------


def apply_action_noise(action: np.ndarray, action_noise: VectorizedActionNoise) -> np.ndarray:
    # We assume that the vectorized action noise uses the same action noise type.
    operation = getattr(action_noise.noises[0], 'operation', '+')
    noise_value = action_noise()
    noise_value = np.squeeze(noise_value, axis=1)
    if operation == '*':
        action *= noise_value
    else:
        action += noise_value

    return action


action_noise_registry = {
    'OrnsteinUhlenbeckActionNoise': noise.OrnsteinUhlenbeckActionNoise,
    'NormalActionNoise': noise.NormalActionNoise,
    'ActionDropOut': ActionDropOut
}


def build_action_noise(config: Config):
    action_noise_config = config.environment.action_noise
    action_noise_class = action_noise_registry[action_noise_config.name]

    return build_from_protobuf(action_noise_class, action_noise_config, to_numpy=True)
