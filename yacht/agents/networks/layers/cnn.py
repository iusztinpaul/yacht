import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import Module


def EIIEDense(filters, activation, kernel_regularizer, weight_decay):
    class _EIIEDense(layers.Layer):
        def __int__(self, name='EIIEDense'):
            super().__init__()

        def build(self, input_shape):
            kernel_width = input_shape[2]

            # TODO: See how to add weight decay
            self.conv_2d = layers.Conv2D(
                filters,
                kernel_size=(1, kernel_width),
                strides=(1, 1),
                padding='valid',
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                # weight_decay=self.kwargs.get('weight_decay')
            )

        def call(self, input_tensor):
            return self.conv_2d(input_tensor)

    return _EIIEDense()


def EIIEOutputWithW(kernel_regularizer=None):
    class _EIIEOutputWithW(layers.Layer):
        def __int__(self, name='EIIEOutputWithW'):
            super().__init__()

        def build(self, input_shape):
            self.batch = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            features = input_shape[3]

            self.reshape_input_tensor = layers.Reshape((height, 1, width * features))
            self.reshape_previous_w = layers.Reshape((height, 1, 1))

            self.conv_2d = layers.Conv2D(
                1,
                kernel_size=(1, 1),
                padding='valid',
                kernel_regularizer=kernel_regularizer,
                # TODO: See how to add weight decay
                # weight_decay=self.kwargs.get('weight_decay')
            )
            self.softmax = layers.Softmax()

        def call(self, input_tensor, previous_w):
            input_tensor = self.reshape_input_tensor(input_tensor)
            previous_w = self.reshape_previous_w(previous_w)

            tensor = tf.concat([input_tensor, previous_w], axis=3)
            # tensor = self.conv_2d(tensor)
            tensor = tensor[:, :, 0, 0]

            # FIXME: Is this really ok ?
            btc_bias = tf.zeros(shape=(self.batch, 1))
            tensor = tf.concat([btc_bias, tensor], axis=1)

            tensor = self.softmax(tensor)

            return tensor

    return _EIIEOutputWithW()
