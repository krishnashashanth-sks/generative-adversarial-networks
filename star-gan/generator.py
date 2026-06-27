from tensorflow.keras import layers,Model
from layers import *
import tensorflow as tf

def build_generator(input_shape=(256,256,3),num_domains=5,gf=64):
  img_input=layers.Input(shape=input_shape) # Use 'shape=' keyword for clarity
  domain_input=layers.Input(shape=(num_domains,))

  # Pass img_input (the tensor) instead of input_shape (the tuple)
  x=ReflectionPadding2D(padding=(3,3))(img_input)
  x=layers.Conv2D(gf,kernel_size=7,strides=1,padding='valid',use_bias=False)(x)
  x=InstanceNormalization()(x) # Using custom InstanceNormalization
  x=layers.ReLU()(x)

  x=layers.Conv2D(gf*2,kernel_size=3,strides=2,padding='same',use_bias=False)(x) # Changed padding to 'same'
  x=InstanceNormalization()(x) # Using custom InstanceNormalization
  x=layers.ReLU()(x)

  x=layers.Conv2D(gf*4,kernel_size=3,strides=2,padding='same',use_bias=False)(x) # Changed padding to 'same'
  x=InstanceNormalization()(x) # Using custom InstanceNormalization
  x=layers.ReLU()(x)

  # Use a Lambda layer to handle dynamic shape operations for tiling domain features
  def tile_domain_features(args):
      features, labels = args
      current_shape = tf.shape(features)
      h = current_shape[1]
      w = current_shape[2]
      # Expand labels to (batch, 1, 1, num_domains) and tile spatially
      expanded_labels = tf.expand_dims(tf.expand_dims(labels, 1), 1)
      tiled_labels = tf.tile(expanded_labels, [1, h, w, 1])
      return tiled_labels

  domain_features_tiled = layers.Lambda(tile_domain_features)([x, domain_input])
  x=layers.Concatenate(axis=-1)([x,domain_features_tiled])

  # Adjust filters for residual blocks to match concatenated feature depth
  for _ in range(9):
    x=residual_block(x,gf*4 + num_domains) # Fixed: filters should include num_domains
  x=layers.Conv2DTranspose(gf*2,kernel_size=4,strides=2,padding='same',use_bias=False)(x)
  x=InstanceNormalization()(x)
  x=layers.ReLU()(x)
  x=layers.Conv2DTranspose(gf,kernel_size=4,strides=2,padding='same',use_bias=False)(x)
  x=InstanceNormalization()(x)
  x=layers.ReLU()(x)
  x=ReflectionPadding2D(padding=(3,3))(x)
  x=layers.Conv2D(input_shape[-1],kernel_size=7,strides=1,padding='valid',use_bias=False,activation='tanh')(x)
  return Model(inputs=[img_input,domain_input],outputs=x,name='Generator')