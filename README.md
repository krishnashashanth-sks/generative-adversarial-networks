# Generative Adversarial Networks (GANs) Collection

Welcome to my repository dedicated to exploring and implementing various types of Generative Adversarial Networks (GANs). This project serves as a practical guide to understanding how different generative models evolve, stabilize, and function.

## 📌 Project Overview
Generative Adversarial Networks are a powerful class of machine learning frameworks where two neural networks—a Generator and a Discriminator—compete against each other in a zero-sum game. This repository translates complex generative architectures into easy-to-understand, clean, and modular Python scripts to study their individual mechanics, loss functions, and spatial generation capabilities.

---

## 🤖 GAN Architectures Included

* **[DCGAN (Deep Convolutional GAN)](./dcgan/):** A powerful architectural evolution utilizing spatial convolutional strides (`nn.ConvTranspose2d` and `nn.Conv2d`), batch normalization layers, and LeakyReLU activations to drastically stabilize structural image synthesis on the MNIST dataset.
* **[VQGAN (Vector Quantized GAN)](./vqgan/):** Combines vector quantization with GANs to learn discrete latent representations, enabling high-quality image generation and manipulation.

* **[StyleGAN-V](./stylegan-v/):** A sophisticated architecture that leverages style-based generation through adaptive instance normalization, enabling fine-grained control over generated image features at multiple scales.

* **[DiscoGAN (Discovery GAN)](./disco-gan/):** Specialized for unsupervised image-to-image translation between different domains, learning to map images from one domain to another without paired examples.

* **[AnimeGAN](./anime-gan/):** A specialized architecture designed for artistic style transfer, transforming real-world images into anime-style artwork while preserving content structure and key features.



---
