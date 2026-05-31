# Deep Convolutional GAN (DCGAN) Implementation

This folder contains a clean, modular, and optimized PyTorch implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the MNIST handwritten digits dataset.

## 🧠 Architectural Highlights
Unlike basic Vanilla GANs that rely on flat, fully-connected layers, DCGAN leverages spatial convolutions to build structural maps. This code strictly adheres to the core guidelines established in the seminal DCGAN paper to stabilize training:

* **Generator Network:** Utilizes fractionally-strided convolutions (`nn.ConvTranspose2d`) to map a 1D latent vector ($z$) into a 2D image matrix. It uses **BatchNorm** and **ReLU** activations, finishing with a **Tanh** output activation layer.
* **Discriminator Network:** Utilizes strided convolutions (`nn.Conv2d`) for spatial downsampling. It replaces pooling layers entirely with convolutions, implements **LeakyReLU** activations, and omits BatchNorm on the very first input layer to allow stable gradient flows.

---

## 📁 Folder File Breakdown
The DCGAN code is split into single-responsibility Python scripts to maximize readability and reusability:

* **`generator.py`**: Declares the `Generator` class, controlling the upsampling structure from latent vector space to final pixel mapping.
* **`discriminator.py`**: Declares the `Discriminator` class, taking an image representation and evaluating a binary probability scalar.
* **`dataset.py`**: Handles downsampling, resizing configurations ($64 \times 64$), normalization distributions ($[-1, 1]$), and PyTorch `DataLoader` spawning.
* **`utils.py`**: Contains background helper functions like the standard custom Gaussian `weights_init`.
* **`train.py`**: Houses the strict mathematical optimization loop containing optimization gradients for both networks.
* **`main.py`**: The central configuration orchestrator that binds the hyperparameters, kicks off training, and plots loss histories.

---

## 🚀 Execution Instructions

Make sure your terminal is positioned at the root level of the main repository (`generative-adversarial-networks/`), then run:

```bash
python dcgan/main.py

```

### Key Training Hyperparameters Configured:

* **Latent Vector Size (`nz`):** 100
* **Base Feature Maps (`ngf` & `ndf`):** 64
* **Learning Rate:** 0.0002
* **Adam Optimizers Beta Coefficients:** $\beta_1 = 0.5$, $\beta_2 = 0.999$
* **Batch Size:** 64

---

## 📊 Performance Monitoring

As training scales through the epochs, you will see a `tqdm` progress bar running in your terminal tracking the relative performance:

* **`Loss_D`**: The overall discriminator error margin (real loss + fake loss optimization metrics).
* **`Loss_G`**: The generator error penalty computed when trying to trick the discriminator.
* **`D(x)` / `D(G(z))**`: The average prediction confidence mapping out the structural evolution.

Periodic progress grids are saved directly to the project's centralized `../outputs/` directory so you can physically watch the generated shapes transition from pure Gaussian noise into fully formed handwritten digits!
