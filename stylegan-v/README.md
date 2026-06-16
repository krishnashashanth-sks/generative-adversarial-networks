# StyleGAN-V for Video Generation

This repository contains a PyTorch implementation of a StyleGAN-V like architecture for high-quality video generation, building upon the principles of StyleGAN for image generation. The project focuses on building the core components from scratch, including custom layers, generator, discriminator, and a training loop.

## Project Structure

This project is organized into several conceptual components, reflecting typical practices in deep learning projects. While presented in a notebook for demonstration, these would ideally reside in separate `.py` files for modularity.

### `dataset.py` (Video Dataset Loading)

Contains the `VideoDataset` class and associated data loading utilities. This component is responsible for:

*   **Loading Video Files**: Efficiently reads video frames from specified paths.
*   **Frame Sampling**: Implements strategies (e.g., random, uniform) to extract a fixed number of frames per video clip.
*   **Preprocessing**: Applies transformations such as resizing, normalization, and data type conversion to prepare video frames for model input.
*   **DataLoader**: Integrates with PyTorch's `DataLoader` to provide batched and shuffled video data for training.

### `discriminator.py` (Discriminator Network)

Defines the `Discriminator` network for the StyleGAN-V model. Key features include:

*   **Progressive Growing**: While not explicitly implemented with progressive training stages, the architecture is designed to handle different resolutions up to a specified `max_res_log2`.
*   **DiscriminatorBlock**: A core building block that includes convolutional layers, spectral normalization, LeakyReLU activations, and downsampling (average pooling).
*   **Minibatch Standard Deviation**: A technique to improve training stability and prevent mode collapse by feeding a statistical summary of the batch to the discriminator.
*   **From RGB**: An initial layer that processes the input video (RGB channels) into the network's internal feature representation.

### `generator.py` (Generator Network)

Defines the `Generator` network, which is responsible for synthesizing video sequences. It consists of:

*   **MappingNetwork**: Maps input latent `z` vectors to disentangled `w` (style) vectors, which control the synthesis process at different layers.
*   **SynthesisBlock**: The main building block of the generator, featuring modulated convolutions (`ModulatedConv3d`), noise injection, and `ScaledLeakyReLU` activations. It supports upsampling for progressive resolution generation.
*   **ModulatedConv3d**: A custom convolutional layer that applies instance-wise style modulation (adaptive instance normalization) to the weights, allowing fine-grained control over features.
*   **NoiseInjection**: Adds per-pixel noise to introduce stochasticity and fine details in the generated output.
*   **SelfAttention3d**: Optionally incorporates 3D self-attention mechanisms within synthesis blocks to capture long-range dependencies in both spatial and temporal dimensions.
*   **Truncation Trick**: Implements the truncation trick in the `forward` pass to improve the perceptual quality of generated videos by sampling `w` closer to the average `mean_w`.
*   **To RGB**: A final layer that maps the internal feature representation to the output RGB video channels.

### `inference.py` (Video Generation / Inference Function)

Provides a utility function `inference` to generate new video samples using a trained `Generator` model. It handles:

*   Setting the generator to evaluation mode (`.eval()`).
*   Disabling gradient computation (`torch.no_grad()`).
*   Generating random latent vectors `z` if not provided.
*   Returning the synthesized video tensor.

### `layers.py` (Custom Layers and Activations)

Contains custom layer implementations essential for StyleGAN-V, including:

*   **PixelNorm**: Normalizes the feature vector on a per-pixel basis.
*   **ScaledLeakyReLU**: A leaky ReLU activation function scaled for stability.
*   **ModulatedConv3d**: Custom 3D convolution with adaptive instance normalization for style modulation.
*   **NoiseInjection**: Module for adding per-pixel learned noise.
*   **SelfAttention3d**: 3D self-attention layer for capturing spatio-temporal dependencies.
*   **MinibatchStdDev**: Adds a channel with the standard deviation of features across the minibatch to aid discriminator training.

### `losses.py` (Loss Functions)

Defines the adversarial and regularization loss functions used during training:

*   **d_logistic_loss**: The non-saturating adversarial loss for the discriminator.
*   **g_logistic_loss**: The non-saturating adversarial loss for the generator.
*   **d_r1_loss**: R1 regularization, a gradient penalty applied to the discriminator's output with respect to real samples, encouraging Lipschitz continuity.

### `main.py` (Main Training Script)

Serves as the entry point for the training process. It orchestrates the overall training flow, including:

*   **Model Initialization**: Instantiating the `Generator` and `Discriminator` models.
*   **Optimizer Setup**: Configuring `Adam` optimizers for both networks.
*   **Data Loading**: Setting up the `VideoDataset` and `DataLoader`.
*   **Training Loop**: Invoking the `train_loop` function for the training iterations.\n*   **Hyperparameter Management**: Manages various training hyperparameters like learning rates, R1 regularization strength, and EMA beta.

### `train.py` (Training Loop Utilities)

Encapsulates the `train_loop` function, which manages the iterative training process. This includes:

*   **Discriminator Update**: Steps for training the discriminator with real and fake videos, including R1 regularization.
*   **Generator Update**: Steps for training the generator to produce more realistic videos.
*   **Mean W Update**: Periodically updates the running average of the `w` latent space (`mean_w`) for the truncation trick, using exponential moving average (EMA).
*   **Progress Reporting**: Prints training progress, loss values, and epoch completion messages.

### `utils.py` (General Utility Functions)

Contains various helper functions that support the main training and model components. Examples from the notebook that would typically reside here include:

*   **update_mean_w**: A dedicated function to manage the exponential moving average update of the generator's `mean_w` vector.
*   Any other small, reusable functions that don't directly fit into a specific model or dataset class, such as plotting utilities or specific data manipulation helpers not covered by `torchvision.transforms`.

## Setup and Usage

1.  **Install Dependencies**: Install required Python packages (e.g., `torch`, `torchvision`, `numpy`, `av`, `gdown`, `unrar`).
2.  **Download Dataset**: Download and extract the UCF101 dataset (or your chosen video dataset).
3.  **Run Training**: Execute the main training script to train the StyleGAN-V model.
4.  **Generate Videos**: Use the `inference` function with the trained generator to create new video samples.
