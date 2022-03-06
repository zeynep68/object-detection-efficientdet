from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Layer, ReLU


class Swish(Layer):
    def __init__(self):
        """ Swish activation (sigmoid weighted linear unit).
         https://arxiv.org/abs/1710.05941
        """
        super(Swish, self).__init__()

    def call(self, x, **kwargs):
        return x * sigmoid(x)


class ReLU6(Layer):
    def __init__(self):
        """ ReLU activation where maximum value is 6."""
        super(ReLU6, self).__init__()

    def __call__(self, x, *args, **kwargs):
        return ReLU(max_value=6)(x)
