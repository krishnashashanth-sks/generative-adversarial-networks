import tensorflow as tf
from tensorflow.keras import layers

class UpsamplingGenerator(tf.keras.Model):
  def __init__(self,output_channels,latent_dim):
    super(UpsamplingGenerator,self).__init__()
    self.latent_dim=latent_dim
    self.latent_features_proj=layers.Dense(128,use_bias=False,input_shape=(latent_dim,))
    self.latent_bn=layers.BatchNormalization()
    self.lrelu=layers.LeakyReLU()
    self.conv1=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',use_bias=False)
    self.bn1=layers.BatchNormalization()
    self.lrelu1=layers.LeakyReLU()
    self.upsample_convt=layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False)
    self.bn2=layers.BatchNormalization()
    self.lrelu2=layers.LeakyReLU()
    self.output_conv=layers.Conv2D(output_channels,(5,5),strides=(1,1),padding='same',activation='tanh')
  def call(self,inputs):
    low_res_image,z=inputs
    batch_size=tf.shape(low_res_image)[0]
    H=tf.shape(low_res_image)[1]
    W=tf.shape(low_res_image)[2]
    latent_processed=self.lrelu(self.latent_bn(self.latent_features_proj(z)))
    latent_tiled=tf.tile(tf.expand_dims(tf.expand_dims(latent_processed,1),1),[1,H,W,1])
    concatenated_features=tf.concat([low_res_image,latent_tiled],axis=-1)
    x=self.lrelu1(self.bn1(self.conv1(concatenated_features)))
    x=self.lrelu2(self.bn2(self.upsample_conv1(x)))
    return self.output_conv(x)