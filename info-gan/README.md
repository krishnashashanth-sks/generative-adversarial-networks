# InfoGAN - Interpretable Representation Learning

## Overview

**Information Maximizing Generative Adversarial Networks (InfoGAN)** is an implementation of InfoGAN that learns interpretable and disentangled representations. This implementation focuses on separating the latent code into:
- **Categorical codes** (c_cat): For discrete factors like digit classes
- **Continuous codes** (c_cont): For continuous factors like rotation and thickness
- **Noise** (z): For unstructured variations

The network learns to map these interpretable codes to meaningful image attributes through mutual information maximization.

## Project Structure

```
info-gan/
├── dataset.py          # Data loading and preprocessing
├── generator.py        # Generator model architecture
├── discriminator.py    # Discriminator model architecture  
├── losses.py          # Loss function definitions
├── train.py           # Training loop
├── main.py            # Main entry point and visualization
└── README.md          # This file
```

## File Descriptions

### `dataset.py`
Handles MNIST dataset loading and preprocessing:
- Resizes images to 28×28
- Normalizes pixel values using [0.5, 0.5]
- Uses DataLoader with batch size 64
- Automatic data download

### `generator.py`
Generator network that synthesizes images from latent codes:
- **Input**: Noise (z), categorical codes (c_cat), continuous codes (c_cont)
- **Architecture**: 
  - Concatenates all inputs
  - Dense layer (256 units) with ReLU
  - Output dense layer with Tanh activation (for normalized images)
- **Output**: Generated fake images

### `discriminator.py`
Discriminator with auxiliary Q-network for latent code prediction:
- **Feature Extraction**: Linear layer (256 units) with LeakyReLU and Dropout (0.3)
- **Three Output Heads**:
  1. **Discriminator head**: Validity prediction (real/fake) with Sigmoid
  2. **Q-network categorical head**: Predicts categorical codes
  3. **Q-network continuous head**: Predicts continuous codes
- **Input**: Flattened images (784 for 28×28)

### `losses.py`
Defines loss functions used during training:
- **BCELoss**: Binary Cross-Entropy for adversarial training
- **CrossEntropyLoss**: For categorical latent code prediction
- **MSELoss**: For continuous latent code prediction

### `train.py`
Main training loop implementing the InfoGAN algorithm:
- **Discriminator Loss**: Combines real/fake classification with Q-network mutual information losses
- **Generator Loss**: Combines adversarial loss with mutual information regularization
- **Lambda (λ)**: Weighting factor for mutual information term (default: 1.0)
- Supports GPU/CPU device handling
- Prints loss metrics every 100 batches

### `main.py`
Entry point and visualization script:
- Configures hyperparameters
- Instantiates models and optimizers
- Runs training loop
- **Visualization**:
  - Varies categorical codes while fixing continuous codes and noise → shows digit disentanglement
  - Varies continuous codes while fixing categorical codes → shows continuous attribute learning (e.g., rotation, thickness)
- Generates and displays grid visualizations

## Hyperparameters

```python
noise_dim = 100                 # Noise vector dimension
latent_dim_categorical = 10    # Number of categorical classes (MNIST digits 0-9)
latent_dim_continuous = 2      # Number of continuous factors
img_dim = 784                   # 28×28 images flattened
batch_size = 64
epochs = 50
learning_rate = 0.0002         # Adam optimizer
betas = (0.5, 0.999)          # Adam momentum parameters
lambda_mi = 1.0               # Mutual information weighting
```

## How to Run

### Prerequisites
```bash
pip install torch torchvision matplotlib tqdm numpy
```

### Training
```bash
python main.py
```

The script will:
1. Download MNIST dataset (if not present)
2. Train the InfoGAN for 50 epochs
3. Generate and display visualization grids showing:
   - Digit variations (categorical disentanglement)
   - Rotation/thickness variations (continuous disentanglement)

## Key Concepts

### Mutual Information Maximization
InfoGAN maximizes mutual information between latent codes (c) and generated images (G(z, c)):

```
L_mi = I(c; G(z, c))
```

This encourages the generator to make latent codes predictable from generated images, forcing the network to learn meaningful representations.

### Disentanglement
- **Categorical codes** learn to represent discrete factors (e.g., which digit to generate)
- **Continuous codes** learn continuous transformations (e.g., rotation, scale, style)
- **Noise** captures remaining stochastic variations

### Q-Network
The discriminator's auxiliary Q-network predicts latent codes from images, providing a supervision signal that encourages disentanglement during training.

## Output

After training completes, the script generates visualization grids:

1. **Categorical Disentanglement**: Shows how varying each categorical code (0-9) while keeping continuous codes fixed produces different digit classes

2. **Continuous Disentanglement**: Shows how varying continuous codes (e.g., rotation from -1 to 1) while keeping a fixed digit produces meaningful transformations

## Architecture Diagram

```
Generator:
  Noise (100) ──┐
  C_cat (10) ───┼─→ Concat ─→ Dense(256) ─→ ReLU ─→ Dense(784) ─→ Tanh ─→ Images
  C_cont (2) ──┘

Discriminator:
  Images (784) ─→ Flatten ─→ Dense(256) ─→ LeakyReLU ─→ Dropout(0.3) ─→ ┬─→ Validity (1)
                                                                          ├─→ Q_cat (10)
                                                                          └─→ Q_cont (2)
```

## References

- **Paper**: InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Networks
  - Authors: Xi Chen, Yan Duan, Ruslan Salakhutdinov, Robert Tran, Ioannis Mitliagkas
  - Link: https://arxiv.org/abs/1606.03657

## Notes

- This implementation uses MNIST for simplicity and fast training
- The network learns interpretable representations without explicit labels for categorical/continuous codes
- Results improve with more epochs, larger capacity models, and tuned hyperparameters
- The implementation includes device handling for GPU/CPU compatibility
