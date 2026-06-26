import tensorflow as tf
from tensorflow.keras import layers

class BaseGenerator(tf.keras.Model):
  def __init__(self,latent_dim):
    super(BaseGenerator,self).__init__()
    self.latent_dim=latent_dim
    self.dense=layers.Dense(7*7*128,use_bias=False,input_shape=(latent_dim,))
    self.bn1=layers.BatchNormalization()
    self.lrelu1=layers.LeakyReLU()
    self.reshape=layers.Reshape((7,7,128))
    self.conv1=layers.Conv2DTranspose(64,(5,5),strides=(1,1),padding='same',use_bias=False)
    self.bn2=layers.BatchNormalization()
    self.lrelu2=layers.LeakyReLU()
    self.output_conv=layers.Conv2D(1,(5,5),strides=(1,1),padding='same',activation='tanh')
  def call(self,inputs):
    x=self.dense(inputs)
    x=self.bn1(x)
    x=self.lrelu1(x)
    x=self.reshape(x)
    x=self.conv1(x)
    x=self.bn2(x)
    x=self.lrelu2(x)
    output_image=self.output_conv(x)
    return output_image