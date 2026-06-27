import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator_wgan(latent_dim):
  model=models.Sequential()
  model.add(layers.Dense(4*4*256,use_bias=False,input_shape=(latent_dim,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Reshape((4,4,256)))
  model.add(layers.Conv2DTranspose(128,(5,5),strides=(2,2),padding='same',use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh'))
  return model