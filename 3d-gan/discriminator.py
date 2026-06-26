import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator_model(input_shape=(64,64,64,1)):
  model=tf.keras.Sequential()
  model.add(layers.Conv3D(64,(4,4,4),strides=(2,2,2),padding='same',input_shape=input_shape))
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))
  assert model.output_shape==(None,input_shape[0]//2,input_shape[1]//2,input_shape[2]//2,64)
  model.add(layers.Conv3D(128,(4,4,4),strides=(2,2,2),padding='same')) # Fixed: Changed Conv2D to Conv3D
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))
  assert model.output_shape==(None,input_shape[0]//4,input_shape[1]//4,input_shape[2]//4,128)
  model.add(layers.Conv3D(256,(4,4,4),strides=(2,2,2),padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())
  model.add(layers.Dropout(0.3))
  assert model.output_shape==(None,input_shape[0]//8,input_shape[1]//8,input_shape[2]//8,256) # Fixed: Changed 128 to 256
  model.add(layers.Flatten()) # Fixed: Changed Flattent() to Flatten()
  model.add(layers.Dense(1,activation='sigmoid'))
  return model