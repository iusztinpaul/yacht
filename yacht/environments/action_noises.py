from stable_baselines3.common import noise

from yacht.config import Config
from yacht.utils import build_from_protobuf

action_noise_registry = {
    'OrnsteinUhlenbeckActionNoise': noise.OrnsteinUhlenbeckActionNoise,
    'NormalActionNoise': noise.NormalActionNoise
}


def build_action_noise(config: Config):
    action_noise_config = config.environment.action_noise
    action_noise_class = action_noise_registry[action_noise_config.name]

    return build_from_protobuf(action_noise_class, action_noise_config, to_numpy=True)
