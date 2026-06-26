import tensorflow as tf
from tensorflow.keras import layers

class BaseDiscriminator(tf.keras.Model):
  def __init__(self):
    super(BaseDiscriminator,self).__init__()
    self.conv1=layers.Conv2D(64,(5,5),strides=(1,1),padding='same',input_shape=(7,7,1))
    self.lrelu1=layers.LeakyReLU()
    self.conv2=layers.Conv2D(128,(5,5),strides=(2,2),padding='same')
    self.bn1=layers.BatchNormalization()
    self.lrelu2=layers.LeakyReLU()
    self.flatten=layers.Flatten()
    self.dense=layers.Dense(1)
  def call(self,inputs):
    x=self.conv1(inputs)
    x=self.lrelu1(x)
    x=self.conv2(x)
    x=self.bn1(x)
    return self.dense(self.flatten(self.lrelu2(x)))