import tensorflow as tf
from tensorflow.keras import layers

class UpsamplingDiscriminator(tf.keras.Model):
  def __init__(self):
    super(UpsamplingDiscriminator,self).__init__()
    self.conv1=layers.Conv2D(64,(5,5),strides=(1,1),padding='same')
    self.lrelu1=layers.LeakyReLU()
    self.conv2=layers.Conv2D(128,(5,5),strides=(2,2),padding='same',use_bias=False)
    self.bn1=layers.BatchNormalization()
    self.lrelu2=layers.LeakyReLU()
    self.conv3=layers.Conv2D(256,(5,5),strides=(2,2),padding='same',use_bias=False)
    self.bn2=layers.BatchNormalization()
    self.lrelu3=layers.LeakyReLU()
    self.flatten=layers.Flatten()
    self.dense=layers.Dense(1)
  def call(self,high_res_image,prev_res_image):
    target_h=tf.shape(prev_res_image)[1]
    target_w=tf.shape(prev_res_image)[2]
    downsampled_high_res=tf.image.resize(
      high_res_image,[target_h,target_w],method=tf.image.ResizeMethod.BICUBIC
    )
    x=tf.concat([downsampled_high_res,prev_res_image],axis=-1)
    return self.dense(self.flatten(self.lrelu3(self.bn2(self.conv3(self.lrelu2(self.bn1(self.conv2(self.lrelu1(self.conv1(x))))))))))