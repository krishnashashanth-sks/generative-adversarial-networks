from tensorflow.keras import layers
from tensorflow import keras

def build_encoder(input_shape,latent_dim):
  encoder_inputs=keras.Input(shape=input_shape)
  x=layers.Conv2D(32,3,activation='relu',strides=2,padding='same')(encoder_inputs) # (14,14)
  x=layers.Conv2D(64,3,activation='relu',strides=2,padding='same')(x) # (7,7)
  x=layers.Conv2D(128,3,activation='relu',strides=1,padding='same')(x) # (7,7) - Adjusted stride for 7x7 feature map
  x=layers.Flatten()(x)
  x=layers.Dense(256,activation='relu')(x)
  z_mean=layers.Dense(latent_dim,name='z_mean')(x)
  z_log_var=layers.Dense(latent_dim,name='z_log_var')(x)
  return keras.Model(encoder_inputs,[z_mean,z_log_var],name='encoder')

def build_decoder(latent_dim,output_shape):
  decoder_inputs=keras.Input(shape=(latent_dim,))
  x=layers.Dense(7*7*128,activation='relu')(decoder_inputs) # Adjusted dense layer size to match encoder output (7x7x128)
  x=layers.Reshape((7,7,128))(x) # Adjusted reshape size
  x=layers.Conv2DTranspose(64,3,activation='relu',strides=2,padding='same')(x) # Upsample to (14,14)
  x=layers.Conv2DTranspose(32,3,activation='relu',strides=2,padding='same')(x) # Upsample to (28,28)
  decoder_outputs=layers.Conv2DTranspose(output_shape[-1],3,activation='sigmoid',padding='same')(x)
  return keras.Model(decoder_inputs,decoder_outputs,name='decoder')

def build_discriminator(input_shape):
  discriminator_inputs=keras.Input(shape=input_shape)
  x=layers.Conv2D(32,3,activation='leaky_relu',strides=2,padding='same')(discriminator_inputs)
  x=layers.Dropout(0.3)(x)
  x=layers.Conv2D(64,3,activation='leaky_relu',strides=2,padding='same')(x)
  x=layers.Dropout(0.3)(x)
  x=layers.Conv2D(128,3,activation='leaky_relu',strides=3,padding='same')(x)
  x=layers.Dropout(0.3)(x)
  x=layers.Flatten()(x)
  discriminator_outputs=layers.Dense(1,activation='sigmoid')(x)
  return keras.Model(discriminator_inputs,discriminator_outputs)