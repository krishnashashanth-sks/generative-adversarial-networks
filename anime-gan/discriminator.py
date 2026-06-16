import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def discriminator_block(filters,size,apply_batchnorm=True):
  initializer=tf.random_normal_initializer(0.,0.02)
  result=keras.Sequential()
  result.add(layers.Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
  if apply_batchnorm:
    result.add(layers.BatchNormalization())
  result.add(layers.LeakyReLU())
  return result

def Discriminator(IMG_WIDTH,IMG_HEIGHT):
  initializer=tf.random_normal_initializer(0.,0.02)
  inp=layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT,3],name='input_image')
  tar=layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT,3],name='target_image')
  x=layers.concatenate([inp,tar])
  down1=discriminator_block(64,4,False)(x)
  down2=discriminator_block(128,4)(down1)
  down3=discriminator_block(256,4)(down2)
  zero_pad1=layers.ZeroPadding2D()(down3)
  conv=layers.Conv2D(512,4,strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)
  batch_norm1=layers.BatchNormalization()(conv)
  leaky_relu=layers.LeakyReLU()(batch_norm1)
  zero_pad2=layers.ZeroPadding2D()(leaky_relu)
  last=layers.Conv2D(1,4,strides=1,kernel_initializer=initializer)(zero_pad2)
  return keras.Model(inputs=[inp,tar],outputs=last)

def GrayDiscriminator(IMG_WIDTH,IMG_HEIGHT):
  initializer=tf.random_normal_initializer(0.,0.02)
  inp=layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT,1],name='input_grayscale_image')
  down1=discriminator_block(64,4,False)(inp)
  down2=discriminator_block(128,4)(down1)
  down3=discriminator_block(256,4)(down2)
  zero_pad1=layers.ZeroPadding2D()(down3)
  conv=layers.Conv2D(512,4,strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)
  batch_norm1=layers.BatchNormalization()(conv)
  leaky_relu=layers.LeakyReLU()(batch_norm1)
  zero_pad2=layers.ZeroPadding2D()(leaky_relu)
  last=layers.Conv2D(1,4,strides=1,kernel_initializer=initializer)(zero_pad2)
  return keras.Model(inputs=inp,outputs=last)