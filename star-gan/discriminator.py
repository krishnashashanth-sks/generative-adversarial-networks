from tensorflow.keras import layers,Model
from layers import *

def build_discriminator(input_shape=(256,256,3),num_domains=5,df=64):
  img_input=layers.Input(shape=input_shape)

  x=layers.Conv2D(df,kernel_size=4,strides=2,padding='same',use_bias=False)(img_input)
  x=layers.LeakyReLU(alpha=0.2)(x)

  x=layers.Conv2D(df*2,kernel_size=4,strides=2,padding='same',use_bias=False)(x)
  x=InstanceNormalization()(x)
  x=layers.LeakyReLU(alpha=0.2)(x)

  x=layers.Conv2D(df*4,kernel_size=4,strides=2,padding='same',use_bias=False)(x)
  x=InstanceNormalization()(x)
  x=layers.LeakyReLU(alpha=0.2)(x)

  x=layers.Conv2D(df*8,kernel_size=4,strides=2,padding='same',use_bias=False)(x)
  x=InstanceNormalization()(x)
  x=layers.LeakyReLU(alpha=0.2)(x)

  x = layers.Conv2D(df*16, kernel_size=4, strides=1, padding='same', use_bias=False)(x)
  x=InstanceNormalization()(x)
  x=layers.LeakyReLU(alpha=0.2)(x)

  src_output=layers.Conv2D(1,kernel_size=4,strides=1,padding='same',activation='sigmoid',name='discriminator_src_output')(x)

  domain_output=layers.GlobalAveragePooling2D()(x)
  domain_output=layers.Dense(num_domains,activation='softmax',name='discriminator_cls_output')(domain_output)

  return Model(inputs=img_input,outputs=[src_output,domain_output],name='Discriminator')