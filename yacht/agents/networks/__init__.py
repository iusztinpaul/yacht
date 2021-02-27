import tensorflow as tf

from agents.networks.modules.cnn import EIIECNN


class Network:
    def __init__(
            self,
            feature_num: int,
            assets_num: int,
            window_size: int,
            layers_config,
            device: str
    ):
        self.feature_num = feature_num
        self.assets_num = assets_num
        self.window_size = window_size
        self.layers_config = layers_config
        self.device = device

        self.eiie_cnn = EIIECNN(
            feature_num=feature_num,
            assets_num=assets_num,
            window_size=window_size,
            layers_config=layers_config,
            device=device
        )
