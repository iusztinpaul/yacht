import tensorflow as tf

from tensorflow.keras import datasets, layers, models


class EIIECNN:
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

        self._network = self.build_network()

    def build_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        return model




