import tensorflow as tf
from tensorflow.keras import layers, Model

def make_generator_model(latent_dim,output_shape=(64,64,64,1)):
  model=tf.keras.Sequential()
  initial_dim=output_shape[0]//8
  # FIX: The Dense layer should output enough units to match the 3D reshape target
  model.add(layers.Dense(initial_dim*initial_dim*initial_dim*512,use_bias=False,input_shape=(latent_dim,))) # Corrected line
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Reshape((initial_dim,initial_dim,initial_dim,512)))
  assert model.output_shape==(None,initial_dim,initial_dim,initial_dim,512)
  model.add(layers.Conv3DTranspose(256,(4,4,4),strides=(2,2,2),padding='same',use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  assert model.output_shape==(None,initial_dim*2,initial_dim*2,initial_dim*2,256)
  model.add(layers.Conv3DTranspose(128,(4,4,4),strides=(2,2,2),padding='same',use_bias=False))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  assert model.output_shape==(None,initial_dim*4,initial_dim*4,initial_dim*4,128)
  model.add(layers.Conv3DTranspose(output_shape[-1],(4,4,4),strides=(2,2,2),padding='same',use_bias=False,activation='tanh'))
  assert model.output_shape==(None,output_shape[0],output_shape[1],output_shape[2],output_shape[3]) # Fixed typo here
  return model