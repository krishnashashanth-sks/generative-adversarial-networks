from tensorflow.keras import models,layers

def build_gan(generator,discriminator):
  discriminator.trainable=False
  gan_input=layers.Input(shape=(generator.input_shape[1],))
  x=generator(gan_input)
  gan_output=discriminator(x)
  model=models.Model(gan_input,gan_output)
  return model