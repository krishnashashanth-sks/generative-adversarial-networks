import tensorflow as tf
class WGAN_GP(tf.keras.Model):
    def __init__(self, generator, critic, latent_dim, gp_weight=10.0, **kwargs):
        super(WGAN_GP, self).__init__(**kwargs)
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

    def compile(self, g_optimizer, c_optimizer, c_steps=5, **kwargs):
        super(WGAN_GP, self).compile(**kwargs)
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.c_steps = c_steps # Number of critic steps per generator step
        self.c_loss_metric = tf.keras.metrics.Mean(name='c_loss')
        self.g_loss_metric = tf.keras.metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [self.c_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated_images = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images)
            predictions = self.critic(interpolated_images, training=True)
        
        gradients = gp_tape.gradient(predictions, [interpolated_images])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # Train the critic
        for _ in range(self.c_steps):
            with tf.GradientTape() as tape:
                noise = tf.random.normal([batch_size, self.latent_dim])
                fake_images = self.generator(noise, training=True)
                
                real_predictions = self.critic(real_images, training=True)
                fake_predictions = self.critic(fake_images, training=True)

                c_loss = tf.reduce_mean(fake_predictions) - tf.reduce_mean(real_predictions)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                c_loss = c_loss + gp * self.gp_weight

            c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradients, self.critic.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            noise = tf.random.normal([batch_size, self.latent_dim])
            fake_images = self.generator(noise, training=True)
            fake_predictions = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_predictions)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        self.c_loss_metric.update_state(c_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"c_loss": self.c_loss_metric.result(), "g_loss": self.g_loss_metric.result()}