# VQGAN Implementation with PyTorch

This repository contains a PyTorch implementation of the Vector Quantized Generative Adversarial Network (VQGAN). VQGAN combines the strengths of VQ-VAE (Vector Quantized Variational Autoencoder) for high-quality image reconstruction and a GAN (Generative Adversarial Network) for generating realistic images. This project aims to provide a clear and modular implementation suitable for research and experimentation.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Features

- **Modular Architecture**: Separate modules for Encoder, Vector Quantizer, Decoder, and Discriminator.
- **Comprehensive Loss Functions**:
    - Reconstruction Loss (L1)
    - Perceptual Loss (using a pre-trained VGG-16)
    - Adversarial Loss (BCEWithLogitsLoss)
    - Quantization Loss
- **Optimizers**: Adam optimizers for both the Generator and Discriminator.
- **Data Loading**: Configurable data loader with transformations (demonstrated with CIFAR-10).
- **Full Training Loop**: Implements the iterative training process for VQGAN.
- **Monitoring & Visualization**: Tools for plotting training losses and visualizing reconstructed images.
- **Model Checkpointing**: Capability to save and load trained model weights.

## Project Structure

The project is organized into several modular Python files to enhance readability and maintainability:

-   `dataset.py`: Contains classes and functions related to data loading, transformations, and dataset preparation (e.g., `get_dataloader`).
-   `train.py`: Implements the core training logic, including the generator and discriminator training steps, and the overall training loop.
-   `utils.py`: Provides utility functions, such as image normalization/unnormalization, plotting tools for losses and reconstructions, and device management.
-   `layers.py`: Defines custom neural network layers or blocks used across the model, such as `ResidualBlock`.
-   `model.py`: Houses the definitions for all VQGAN components: `Encoder`, `VectorQuantizer`, `Decoder`, `Discriminator`, and the main `VQGAN` class.
-   `losses.py`: Defines the various loss functions used in VQGAN training, including `VGGPerceptualLoss`.
-   `main.py`: The entry point for the application, handling argument parsing, model initialization, training initiation, and result visualization.

## Setup

1.  **Clone the repository** (if applicable):
    ```bash
    git clone https://github.com/krishnashashanth-sks/vqgan-pytorch.git
    cd vqgan-pytorch
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install torch torchvision matplotlib tqdm
    ```

## Usage

To train the VQGAN model, run the `main.py` script:

```bash
python main.py
```

You can customize training parameters using command-line arguments. Refer to `main.py` for all available options.

## Training

The `train.py` module orchestrates the training process. During training, the VQGAN attempts to reconstruct input images while the Discriminator distinguishes between real images and generated reconstructions. The Perceptual Loss, using VGG features, guides the generator to produce visually similar outputs.

Losses will be logged to the console, and plots of generator/discriminator losses and sample reconstructions will be saved or displayed upon completion of training.

## Results

After training, you can expect:
-   Plots showing the progression of generator and discriminator losses over batches/epochs.
-   Visualizations of real images alongside their VQGAN reconstructions, demonstrating the model's ability to capture image details.
