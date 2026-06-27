from tensorflow.keras import layers
from utils import reflect_pad
import tensorflow as tf

class ReflectionPadding2D(layers.Layer):
  def __init__(self, padding=(1, 1), **kwargs):
    super(ReflectionPadding2D, self).__init__(**kwargs)
    if isinstance(padding, int):
        self.padding = (padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 2:
        self.padding = padding
    else:
        raise ValueError(f"Padding must be an int or a tuple of 2 ints. Received: {padding}")

  def call(self, inputs):
    # Assumes padding[0] is for height and width. If separate, this needs adjustment.
    return reflect_pad(inputs, self.padding[0])

  def get_config(self):
    config = super(ReflectionPadding2D, self).get_config()
    config.update({
        'padding': self.padding
    })
    return config
class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        normalized_x = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized_x + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config
    
def residual_block(x, filters, kernel_size=3, stride=1, use_bias=False):
  x_skip = x
  # First Conv Block
  x = ReflectionPadding2D(padding=(1, 1))(x)
  x = layers.Conv2D(filters, kernel_size, strides=stride, padding='valid', use_bias=use_bias)(x)
  x = InstanceNormalization()(x) # Using custom InstanceNormalization
  x = layers.ReLU()(x)

  # Second Conv Block
  x = ReflectionPadding2D(padding=(1, 1))(x)
  x = layers.Conv2D(filters, kernel_size, strides=stride, padding='valid', use_bias=use_bias)(x)
  x = InstanceNormalization()(x) # Using custom InstanceNormalization

  x = layers.Add()([x, x_skip])
  return x