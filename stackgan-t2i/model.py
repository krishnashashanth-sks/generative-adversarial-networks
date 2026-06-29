import  os
from tensorflow.keras.optimizers import Adam
from utils import *
from layers import *
import time

class StackGANStage1(object):
  def __init__(self, epochs=100, x_dim=100, batch_size=64,
               stage1_generator_lr=0.0002, stage1_discriminator_lr=0.0002,
               data_dir="/content/drive/MyDrive/data/birds/birds"):
    self.epochs = epochs
    self.x_dim = x_dim # Noise dimension
    self.batch_size = batch_size

    # Dataset paths (dynamically set based on data_dir)
    self.data_dir = data_dir
    self.train_dir = os.path.join(self.data_dir, "train")
    self.class_id_path_train = os.path.join(self.train_dir, "class_info.pickle")
    self.filename_path_train = os.path.join(self.train_dir, "filenames.pickle")
    self.embedding_dir_train = os.path.join(self.train_dir, "char-CNN-RNN-embeddings.pickle")
    # This path refers to the root of the raw CUB dataset (e.g., CUB_200_2011/CUB_200_2011) from which images are loaded
    self.dataset_root_path = os.path.join(self.data_dir, "CUB_200_2011", "CUB_200_2011")

    self.image_size = 64 # For Stage-I GAN
    self.conditioning_dim = 128 # Output dimension of 'c' from CA

    # Optimizers
    self.stage1_generator_optimizer = Adam(learning_rate=stage1_generator_lr, beta_1=0.5, beta_2=0.999)
    self.stage1_discriminator_optimizer = Adam(learning_rate=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)

    # Build Models
    self.embedding_compressor = build_embedding_compressor()

    self.stage1_generator = build_stage1_generator()
    self.stage1_discriminator = build_stage1_discriminator(image_size=self.image_size, conditioning_dim=self.conditioning_dim)

    self.stage1_discriminator.compile(loss='binary_crossentropy', optimizer=self.stage1_discriminator_optimizer)

    self.stage1_adversarial = build_adversarial(self.stage1_generator, self.stage1_discriminator)
    self.stage1_adversarial.compile(
        loss=['binary_crossentropy', adversarial_loss], # First for D's prob, second for KL divergence on MLS
        loss_weights=[1.0, 2.0], # Weights for the two losses
        optimizer=self.stage1_generator_optimizer # Generator trains adversarial model
    )

    # Checkpoints
    self.checkpoint = tf.train.Checkpoint(
        generator_optimizer=self.stage1_generator_optimizer,
        discriminator_optimizer=self.stage1_discriminator_optimizer,
        generator=self.stage1_generator,
        discriminator=self.stage1_discriminator
    )
    self.checkpoint_prefix = os.path.join("./checkpoints_stage1", "ckpt")
    os.makedirs("./checkpoints_stage1", exist_ok=True)
    os.makedirs("generated_images_stage1", exist_ok=True)


  def train_stage1(self):
    print("Loading Stage-I training data...")
    x_train, y_train, train_embeds = load_data(
        class_id_path=self.class_id_path_train,
        filename_path=self.filename_path_train,
        embeddings_path=self.embedding_dir_train,
        size=self.image_size,
        dataset_root_path=self.dataset_root_path
    )
    if x_train.size == 0:
      print("No training data loaded for Stage 1. Aborting training.")
      return

    print(f"Loaded {x_train.shape[0]} training images for Stage I.")

    # Normalize images to [-1, 1]
    x_train = (x_train - 127.5) / 127.5

    real_labels = np.ones((self.batch_size, 1), dtype='float32') * 0.9 # Label smoothing
    fake_labels = np.zeros((self.batch_size, 1), dtype='float32') + 0.1 # Label smoothing

    for epoch in range(self.epochs):
      start_time = time.time()
      print(f"\nEpoch: {epoch}")

      # Shuffle data before each epoch
      indices = np.arange(x_train.shape[0])
      np.random.shuffle(indices)
      x_train_shuffled = x_train[indices]
      train_embeds_shuffled = train_embeds[indices]

      num_batches = x_train.shape[0] // self.batch_size

      for i in range(num_batches):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)

        image_batch_normalized = x_train_shuffled[idx]
        embedding_batch_raw = train_embeds_shuffled[idx]
        latent_space = np.random.normal(0, 1, size=(self.batch_size, self.x_dim))

        # Get conditional embedding 'c' (128-dim) from the raw text embedding batch for Discriminator conditioning
        c_embeddings_raw = self.embedding_compressor.predict_on_batch(embedding_batch_raw) # 128-dim

        # --- Train Discriminator ---
        # Train on Real images + correct conditional embedding
        d_loss_real = self.stage1_discriminator.train_on_batch(
            [image_batch_normalized, c_embeddings_raw], real_labels
        )

        # Generate fake images and their conditional embeddings
        gen_images, c_fake_for_dis, mls_fake = self.stage1_generator.predict_on_batch([embedding_batch_raw, latent_space])

        # Train on Fake images + correct conditional embedding
        d_loss_fake = self.stage1_discriminator.train_on_batch(
            [gen_images, c_fake_for_dis], fake_labels
        )

        # Train on Wrong (Real image with mismatched conditional embedding) -- Optional but good for stability
        # Shift embeddings to create mismatched pairs
        wrong_c_embeddings_raw = np.roll(c_embeddings_raw, 1, axis=0)
        d_loss_wrong = self.stage1_discriminator.train_on_batch(
            [image_batch_normalized, wrong_c_embeddings_raw], fake_labels
        )

        d_loss = 0.5 * np.add(d_loss_real, 0.5 * np.add(d_loss_fake, d_loss_wrong))

        # --- Train Generator (Adversarial) ---
        # G wants D to classify fakes as real, and also minimizes KL divergence of CA
        g_loss = self.stage1_adversarial.train_on_batch(
            [embedding_batch_raw, latent_space],
            [real_labels, mls_fake] # Target for D's output (real_labels) and target for KL loss (mls_fake)
        )

        if i % 100 == 0:
            print(f"Batch {i}/{num_batches} - D_loss: {d_loss:.4f}, G_loss_total: {g_loss[0]:.4f}, G_loss_adv: {g_loss[1]:.4f}, G_loss_kl: {g_loss[2]:.4f}")

      # Save Checkpoints
      if (epoch + 1) % 15 == 0:
          self.checkpoint.save(file_prefix=self.checkpoint_prefix)
          print(f"Checkpoint saved at epoch {epoch+1}")

      # Generate and save sample images for visualization
      if (epoch + 1) % 5 == 0:
          fixed_embedding_batch = train_embeds_shuffled[0:self.batch_size] # Use a fixed set of embeddings
          fixed_noise = np.random.normal(0, 1, size=(self.batch_size, self.x_dim))

          generated_images, _, _ = self.stage1_generator.predict_on_batch([fixed_embedding_batch, fixed_noise])

          # Denormalize images for saving
          generated_images = (generated_images * 127.5) + 127.5
          generated_images = generated_images.astype(np.uint8)

          # Save the first generated image of the batch
          save_path = os.path.join("generated_images_stage1", f"epoch_{epoch+1}_generated_image.png")
          save_image(generated_images[0], save_path)
          print(f"Sample image saved to {save_path}")

      print(f"Epoch {epoch} took {time.time() - start_time:.2f} seconds")


class StackGANStage2(object):
    def __init__(self, epochs=50, x_dim=100, batch_size=64,
                 stage2_generator_lr=0.0002, stage2_discriminator_lr=0.0002,
                 data_dir="/content/drive/MyDrive/data/birds/birds"):
        self.epochs = epochs
        self.x_dim = x_dim
        self.batch_size = batch_size

        # Dataset paths
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, "train")
        self.class_id_path_train = os.path.join(self.train_dir, "class_info.pickle")
        self.filename_path_train = os.path.join(self.train_dir, "filenames.pickle")
        self.embedding_dir_train = os.path.join(self.train_dir, "char-CNN-RNN-embeddings.pickle")
        self.dataset_root_path = os.path.join(self.data_dir, "CUB_200_2011", "CUB_200_2011")

        self.low_image_size = 64
        self.high_image_size = 256
        self.conditioning_dim = 128

        self.stage2_generator_optimizer = Adam(learning_rate=stage2_generator_lr, beta_1=0.5, beta_2=0.999)
        self.stage2_discriminator_optimizer = Adam(learning_rate=stage2_discriminator_lr, beta_1=0.5, beta_2=0.999)

        # Build Models (S1 Gen is needed for S2 Adv)
        self.stage1_generator = build_stage1_generator() # Re-initialize S1 Gen to pass to S2 Adv
        self.stage2_generator = build_stage2_generator()
        self.stage2_discriminator = build_stage2_discriminator(image_size=self.high_image_size, conditioning_dim=self.conditioning_dim)

        self.stage2_generator.compile(loss='binary_crossentropy', optimizer=self.stage2_generator_optimizer) # This compilation might not be strictly necessary if trained via adversarial
        self.stage2_discriminator.compile(loss='binary_crossentropy', optimizer=self.stage2_discriminator_optimizer)

        self.stage2_adversarial = build_stage2_adversarial(self.stage2_discriminator, self.stage2_generator, self.stage1_generator)
        self.stage2_adversarial.compile(loss=['binary_crossentropy', adversarial_loss], loss_weights=[1.0, 2.0], optimizer=self.stage2_generator_optimizer)

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.stage2_generator_optimizer,
            discriminator_optimizer=self.stage2_discriminator_optimizer,
            generator_stage1=self.stage1_generator,
            generator_stage2=self.stage2_generator,
            discriminator_stage2=self.stage2_discriminator
        )
        self.checkpoint_prefix = os.path.join("./checkpoints_stage2", "ckpt")
        os.makedirs("./checkpoints_stage2", exist_ok=True)
        os.makedirs("generated_images_stage2", exist_ok=True)

    def train_stage2(self):
        print("Loading Stage-II training data...")
        x_high_train, _, high_train_embeds = load_data(
            class_id_path=self.class_id_path_train,
            filename_path=self.filename_path_train,
            embeddings_path=self.embedding_dir_train,
            size=self.high_image_size,
            dataset_root_path=self.dataset_root_path
        )
        if x_high_train.size == 0:
          print("No training data loaded for Stage 2. Aborting training.")
          return

        print(f"Loaded {x_high_train.shape[0]} training images for Stage II.")

        x_high_train = (x_high_train - 127.5) / 127.5

        real_labels = np.ones((self.batch_size, 1), dtype='float32') * 0.9
        fake_labels = np.zeros((self.batch_size, 1), dtype='float32') + 0.1

        for epoch in range(self.epochs):
            start_time = time.time()
            print(f"\nEpoch: {epoch}")

            indices = np.arange(x_high_train.shape[0])
            np.random.shuffle(indices)
            x_high_train_shuffled = x_high_train[indices]
            high_train_embeds_shuffled = high_train_embeds[indices]

            num_batches = x_high_train.shape[0] // self.batch_size

            for i in range(num_batches):
                idx = slice(i * self.batch_size, (i + 1) * self.batch_size)

                image_batch_high_res_normalized = x_high_train_shuffled[idx]
                embedding_batch_raw = high_train_embeds_shuffled[idx]
                latent_space = np.random.normal(0, 1, size=(self.batch_size, self.x_dim))

                # Generate low-res image using Stage-I Generator (this is an inferential step)
                # Note: stage1_generator.predict returns [image, c_embedding, mls]
                low_res_fake, c_embedding_s1, mls_s1 = self.stage1_generator.predict_on_batch([embedding_batch_raw, latent_space])

                # Generate high-res image using Stage-II Generator (also inferential for D training)
                # Note: stage2_generator.predict returns [image, c_embedding, mls]
                high_res_fake, c_embedding_s2, mls_s2 = self.stage2_generator.predict_on_batch([embedding_batch_raw, low_res_fake])

                # --- Train Discriminator ---
                d_loss_real = self.stage2_discriminator.train_on_batch(
                    [image_batch_high_res_normalized, c_embedding_s2], real_labels
                )

                d_loss_fake = self.stage2_discriminator.train_on_batch(
                    [high_res_fake, c_embedding_s2], fake_labels
                )

                # Train on Wrong (Real high-res image with mismatched conditional embedding)
                wrong_c_embedding = np.roll(c_embedding_s2, 1, axis=0)
                d_loss_wrong = self.stage2_discriminator.train_on_batch(
                    [image_batch_high_res_normalized, wrong_c_embedding], fake_labels
                )

                d_loss = 0.5 * np.add(d_loss_real, 0.5 * np.add(d_loss_fake, d_loss_wrong))

                # --- Train Generator (Adversarial) ---
                # G2 wants D2 to classify its fakes as real, and also minimizes KL divergence of CA
                g_loss = self.stage2_adversarial.train_on_batch(
                    [embedding_batch_raw, latent_space],
                    [real_labels, mls_s2] # Target for D's output (real_labels) and target for KL loss (mls_s2)
                )

                if i % 100 == 0:
                    print(f"Batch {i}/{num_batches} - D_loss: {d_loss:.4f}, G_loss_total: {g_loss[0]:.4f}, G_loss_adv: {g_loss[1]:.4f}, G_loss_kl: {g_loss[2]:.4f}")

            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"Checkpoint saved at epoch {epoch+1}")

            if (epoch + 1) % 5 == 0:
                fixed_embedding_batch = high_train_embeds_shuffled[0:self.batch_size] # Use a fixed set of embeddings
                fixed_latent_space = np.random.normal(0, 1, size=(self.batch_size, self.x_dim))

                # Generate low-res image for visualization
                low_res_viz, _, _ = self.stage1_generator.predict_on_batch([fixed_embedding_batch, fixed_latent_space])
                # Generate high-res image for visualization
                high_res_viz, _, _ = self.stage2_generator.predict_on_batch([fixed_embedding_batch, low_res_viz])

                high_res_viz = (high_res_viz * 127.5) + 127.5
                high_res_viz = high_res_viz.astype(np.uint8)

                save_path = os.path.join("generated_images_stage2", f"epoch_{epoch+1}_generated_image_256.png")
                save_image(high_res_viz[0], save_path)
                print(f"Sample 256x256 image saved to {save_path}")

            print(f"Epoch {epoch} took {time.time() - start_time:.2f} seconds")
