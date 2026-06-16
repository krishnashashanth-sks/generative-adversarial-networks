# Pix2Pix GAN - Image-to-Image Translation

A TensorFlow implementation of Pix2Pix (Conditional Generative Adversarial Networks) for paired image-to-image translation tasks. This project demonstrates how to train a conditional GAN to learn a mapping from input images to output images.

## Project Overview

Pix2Pix is a conditional GAN framework that learns a mapping from an input image to an output image using paired examples. It's widely used for applications like:
- Photo enhancement
- Semantic segmentation visualization
- Style transfer
- Image colorization
- Architecture reconstruction

## Project Structure

```
pix2pix-gan/
├── main.py              # Entry point - Initializes and runs the training pipeline
├── train.py             # Training loop and train_step function
├── generator.py         # Generator model (U-Net architecture)
├── discriminator.py     # Discriminator model (PatchGAN)
├── losses.py            # Loss functions for generator and discriminator
├── optimizers.py        # Optimizer configurations
├── dataset.py           # Data loading and preprocessing
├── inference.py         # Inference and image generation
└── utils.py             # Utility functions
```

## File Descriptions

### `main.py`
The entry point of the training pipeline. It:
- Creates and summarizes the Generator and Discriminator models
- Loads training and test datasets
- Initiates the training process
- Performs inference on test examples
- Generates and saves output images

**Key Parameters:**
- `NUM_EPOCHS`: Number of training epochs (default: 5)
- `OUTPUT_CHANNELS`: Number of output image channels (default: 3 for RGB)

### `train.py`
Contains the core training logic:
- `train_step()`: Performs a single training step for both generator and discriminator
  - Computes generator and discriminator losses
  - Calculates gradients
  - Updates model weights
- `fit()`: Trains the models over multiple epochs

### `generator.py`
Implements the Generator model using U-Net architecture:
- **Encoder (Downsampling)**: 8 downsampling layers that reduce spatial dimensions
- **Decoder (Upsampling)**: 7 upsampling layers with skip connections
- **Output Layer**: Conv2DTranspose layer with tanh activation

Features:
- Input shape: [256, 256, 3]
- Downsampling filters: 64 → 512
- Skip connections between encoder and decoder
- Dropout in upsampling layers for regularization

### `discriminator.py`
Implements the Discriminator model (PatchGAN):
- Classifies 70×70 patches of the image as real or fake
- Takes both input and target/generated images as input
- Uses convolutional layers with batch normalization and LeakyReLU

### `losses.py`
Defines loss functions:
- `generator_loss()`: Combines adversarial loss and L1 reconstruction loss
- `discriminator_loss()`: Binary crossentropy loss for real vs. generated images

### `optimizers.py`
Configures optimizers:
- Uses Adam optimizer for both generator and discriminator
- Custom learning rates and beta values

### `dataset.py`
Handles data loading and preprocessing:
- Loads image pairs from the dataset
- Performs image normalization and resizing
- Creates training and test datasets
- Applies data augmentation

### `inference.py`
Generates predictions and saves results:
- Takes generator, input image, and target image
- Produces generated output
- Saves comparison images

### `utils.py`
Utility functions for:
- Image preprocessing and postprocessing
- File I/O operations
- Visualization helpers

## Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)

### Installation

```bash
# Clone the repository
git clone https://github.com/krishnashashanth-sks/generative-adversarial-networks.git
cd generative-adversarial-networks/pix2pix-gan

# Install required dependencies
pip install tensorflow numpy matplotlib
```

### Running the Training

```bash
python main.py
```

This will:
1. Create the generator and discriminator models
2. Load the training dataset
3. Train for the specified number of epochs
4. Generate sample outputs from the test set

### Model Architecture

**Generator (U-Net):**
```
Input: [256, 256, 3]
  ↓
Encoder: 8 downsampling blocks
  ↓
Decoder: 7 upsampling blocks with skip connections
  ↓
Output: [256, 256, 3]
```

**Discriminator (PatchGAN):**
```
Input: ([256, 256, 3], [256, 256, 3])  # input image + target/generated
  ↓
5 convolutional blocks
  ↓
Output: [1]  # Real or Fake classification
```

## Training Details

- **Loss Function**: Generator loss combines:
  - Adversarial loss (makes discriminator fooled)
  - L1 reconstruction loss (keeps output similar to target)
- **Optimizer**: Adam optimizer with β₁ = 0.5, β₂ = 0.999
- **Learning Rate**: 2e-4
- **Batch Size**: Configurable in dataset.py

## Results

After training, the model generates images and saves them for comparison with target images. The quality improves with:
- More training epochs
- Larger and more diverse datasets
- Fine-tuning hyperparameters

## References

- [Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)](https://arxiv.org/abs/1611.05957)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [PatchGAN Discriminators](https://github.com/phillipi/pix2pix)

## License

This project is part of the Generative Adversarial Networks repository.

## Contributing

Feel free to fork this project, make improvements, and submit pull requests!
