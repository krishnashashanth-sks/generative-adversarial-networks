from tensorflow.keras import optimizers
import tensorflow as tf

def train_gan(generator,discriminator,gan,dataset,latent_dim,epochs=100,batch_size=128):
  discriminator_optimizer=optimizers.Adam(learning_rate=0.0003,beta_1=0.5)
  generator_optimizer=optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=False)
  discriminator.compile(optimizer=discriminator_optimizer,loss=loss_fn,metrics=['accuracy'])
  gan.compile(optimizer=generator_optimizer,loss=loss_fn)
  for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for batch in dataset:
      noise=tf.random.normal([batch_size,latent_dim])
      generated_images=generator(noise,training=True)
      real_images=batch
      combined_images=tf.concat([real_images,generated_images],axis=0)
      labels=tf.concat([tf.ones((real_images.shape[0],1)),tf.zeros((generated_images.shape[0],1))],axis=0)
      labels+=0.05*tf.random.uniform(labels.shape)
      d_loss=discriminator.train_on_batch(combined_images,labels)
      noise=tf.random.normal([batch_size,latent_dim])
      misleading_labels=tf.ones([batch_size,1])
      g_loss=gan.train_on_batch(noise,misleading_labels)
    print(f"Discriminator loss:{d_loss[0]:.4f}.Accuracy:{d_loss[1]:.4f}")
    print(f"  Generator Loss: {g_loss:.4f}")