import tensorflow as tf
from tensorflow import keras

class VAEGAN(keras.Model):
  def __init__(self,encoder,decoder,discriminator,lambda_rec,latent_dim,lambda_kl,lambda_adv,**kwargs):
    super().__init__(**kwargs)
    self.encoder=encoder
    self.decoder=decoder
    self.discriminator=discriminator
    self.lambda_rec=lambda_rec
    self.latent_dim=latent_dim
    self.lambda_kl=lambda_kl
    self.lambda_adv=lambda_adv
    self.total_loss_tracker=keras.metrics.Mean(name='total_loss')
    self.reconstruction_loss_tracker=keras.metrics.Mean(name='reconstruction_loss')
    self.kl_loss_tracker=keras.metrics.Mean(name='kl_loss')
    self.generator_loss_tracker=keras.metrics.Mean(name='generator_loss')
    self.discriminator_loss_tracker=keras.metrics.Mean(name='discriminator_loss')
  def call(self,inputs):
    z_mean,z_log_var=self.encoder(inputs)
    batch=tf.shape(z_mean)[0]
    dim=tf.shape(z_mean)[1]
    epsilon=tf.keras.backend.random_normal(shape=(batch,dim))
    z=z_mean+tf.exp(0.5*z_log_var)*epsilon
    reconstructed_images=self.decoder(z)
    return reconstructed_images,z_mean,z_log_var
  def compile(self,vae_optimizer,discriminator_optimizer,**kwargs):
    super().compile(**kwargs)
    self.vae_optimizer=vae_optimizer
    self.discriminator_optimizer=discriminator_optimizer
    self.reconstruction_loss_fn=keras.losses.BinaryCrossentropy(from_logits=False)
    self.discriminator_loss_fn=keras.losses.BinaryCrossentropy(from_logits=False)
  @property
  def metrics(self):
    return[
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker,
        self.generator_loss_tracker,
        self.discriminator_loss_tracker,
    ]
  def train_step(self,data):
    real_images=data
    with tf.GradientTape()as tape:
      z_mean,z_log_var=self.encoder(real_images)
      batch=tf.shape(z_mean)[0]
      dim=tf.shape(z_mean)[1]
      epsilon=tf.keras.backend.random_normal(shape=(batch,dim))
      z_reconstruction=z_mean+tf.exp(0.5*z_log_var)*epsilon
      reconstructed_images=self.decoder(z_reconstruction)
      random_latent_vectors=tf.keras.backend.random_normal(shape=(batch,self.latent_dim))
      generated_images=self.decoder(random_latent_vectors)
      real_predictions=self.discriminator(real_images)
      reconstructed_predictions=self.discriminator(reconstructed_images)
      generated_predictions=self.discriminator(generated_images)
      real_labels=tf.ones_like(real_predictions)
      fake_labels=tf.zeros_like(generated_predictions)
      d_loss_real=self.discriminator_loss_fn(real_labels,real_predictions)
      d_loss_reconstruted=self.discriminator_loss_fn(fake_labels,reconstructed_predictions)
      d_loss_generated=self.discriminator_loss_fn(fake_labels,generated_predictions)
      d_loss=(d_loss_real+d_loss_reconstruted+d_loss_generated)/3
    d_grad=tape.gradient(d_loss,self.discriminator.trainable_weights)
    self.discriminator_optimizer.apply_gradients(zip(d_grad,self.discriminator.trainable_weights))
    with tf.GradientTape() as tape:
      z_mean,z_log_var=self.encoder(real_images)
      batch=tf.shape(z_mean)[0]
      dim=tf.shape(z_mean)[1]
      epsilon=tf.keras.backend.random_normal(shape=(batch,dim))
      z=z_mean+tf.exp(0.5*z_log_var)*epsilon
      reconstructed_images,_,_=self.call(real_images)
      reconstrution_loss=self.reconstruction_loss_fn(real_images,reconstructed_images)
      kl_loss=-0.5*tf.reduce_sum(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var),axis=1)
      kl_loss=tf.reduce_mean(kl_loss)
      reconstructed_predictions=self.discriminator(reconstructed_images)
      generated_adversarial_loss=self.discriminator_loss_fn(tf.ones_like(reconstructed_predictions),reconstructed_predictions)
      total_vae_loss=self.lambda_rec*reconstrution_loss+self.lambda_kl*kl_loss+self.lambda_adv*generated_adversarial_loss
    vae_grads=tape.gradient(total_vae_loss,self.encoder.trainable_weights+self.decoder.trainable_weights)
    self.vae_optimizer.apply_gradients(zip(vae_grads,self.encoder.trainable_weights+self.decoder.trainable_weights))
    self.total_loss_tracker.update_state(total_vae_loss+d_loss)
    self.reconstruction_loss_tracker.update_state(reconstrution_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    self.generator_loss_tracker.update_state(generated_adversarial_loss)
    self.discriminator_loss_tracker.update_state(d_loss)
    return {
        "total_loss":self.total_loss_tracker.result(),
        "reconstruction_loss":self.reconstruction_loss_tracker.result(),
        "kl_loss":self.kl_loss_tracker.result(),
        "generator_loss":self.generator_loss_tracker.result(),
        "discriminator_loss":self.discriminator_loss_tracker.result()
    }