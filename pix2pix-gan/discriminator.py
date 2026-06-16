import tensorflow as tf
from generator import downsample

def Discriminator():
  initializer=tf.random_normal_initializer(0.,0.02)
  inp=tf.keras.layers.Input(shape=[256,256,3],name='input_image')
  tar=tf.keras.layers.Input(shape=[256,256,3],name='target_image')
  x=tf.keras.layers.concatenate([inp,tar])
  down1=downsample(64,4,False)(x)
  down2=downsample(128,4)(down1)
  down3=downsample(256,4)(down2)
  zero_pad1=tf.keras.layers.ZeroPadding2D()(down3)
  conv=tf.keras.layers.Conv2D(512,4,strides=1,kernel_initializer=initializer,use_bias=False)(zero_pad1)
  batch_norm1=tf.keras.layers.BatchNormalization()(conv)
  leaky_relu=tf.keras.layers.LeakyReLU()(batch_norm1)
  zero_pad2=tf.keras.layers.ZeroPadding2D()(leaky_relu)
  last=tf.keras.layers.Conv2D(1,4,strides=1,kernel_initializer=initializer)(zero_pad2)
  return tf.keras.Model(inputs=[inp,tar],outputs=last)