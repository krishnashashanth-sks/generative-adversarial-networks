import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def downsample(filters,size,apply_batchnorm=True):
  initializer=tf.random_normal_initializer(0.,0.02)
  result=keras.Sequential()
  result.add(layers.Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
  if apply_batchnorm:
    result.add(layers.BatchNormalization())
  result.add(layers.LeakyReLU())
  return 
  
def upsample(filters,size,apply_dropout=False):
  initializer=tf.random_normal_initializer(0.,0.02)
  result=keras.Sequential()
  result.add(layers.Conv2DTranspose(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
  if apply_dropout:
    result.add(layers.Dropout(0.5))
  result.add(layers.ReLU())
  return result

def Generator(IMG_WIDTH,IMG_HEIGHT):
  inputs=layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT,3])
  down_stack=[
      downsample(64,4,apply_batchnorm=False),
      downsample(128,4),
      downsample(256,4),
      downsample(512,4),
      downsample(512,4),
      downsample(512,4),
      downsample(512,4),
      downsample(512,4),
  ]
  up_stack=[
      upsample(512,4,apply_dropout=True),
      upsample(512,4,apply_dropout=True),
      upsample(512,4,apply_dropout=True),
      upsample(512,4),
      upsample(256,4),
      upsample(128,4),
      upsample(64,4),
  ]
  initializer=tf.random_normal_initializer(0.,0.02)
  last=layers.Conv2DTranspose(filters=3,kernel_size=4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh')
  x=inputs
  skips=[]
  for down in down_stack:
    x=down(x)
    skips.append(x)
  skips=skips[:-1]
  skips=reversed(skips)
  for up,skip in zip(up_stack,skips):
    x=up(x)
    x=layers.Concatenate()([x,skip])
  x=last(x)
  return keras.Model(inputs=inputs,outputs=x)