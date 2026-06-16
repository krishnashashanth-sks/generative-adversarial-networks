# Discovery GAN (DiscoGAN) for Unsupervised Image-to-Image Translation

## Introduction

This project implements a Discovery Generative Adversarial Network (DiscoGAN) for unsupervised image-to-image translation. It enables learning mappings between two distinct image domains (e.g., MNIST to SVHN) without requiring paired training examples. The architecture relies on cycle-consistency and adversarial losses to achieve stable and high-quality image transformations.

## Project Structure

The project is organized into several Python files to manage different aspects of the DiscoGAN implementation:

-   `main.py`: Contains the main training loop, orchestrates data loading, model initialization, and training process.
-   `dataset.py`: Handles dataset loading (MNIST, SVHN), image transformations, and creation of PyTorch `DataLoader` objects.
-   `generator.py`: Defines the `Generator` (U-Net-like) network architecture.
-   `discriminator.py`: Defines the `Discriminator` (PatchGAN-like) network architecture.
-   `losses.py`: Defines the loss functions used in the GAN (adversarial, cycle, identity).
-   `train.py`: Encapsulates the training logic for generators and discriminators.
-   `inference.py`: Provides utilities for performing and visualizing image translation on new inputs.
-   `utils.py`: Includes various utility functions such as image visualization, loss plotting, and helper functions.
-   `requirements.txt`: Lists all necessary Python dependencies.
-   `README.md`: This file, providing an overview and instructions.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/krishnashashanth-sks/generative-adversarial-networks
    cd generative-adversarial-networks/disco-gan
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

To train the Discovery GAN:

1.  **Ensure all dependencies are installed.**
2.  **Execute the main training script:**
    ```bash
    python main.py
    ```
    The `main.py` script will:
    *   Download the MNIST and SVHN datasets (if not already present).
    *   Initialize and train the Generator and Discriminator models.
    *   Save generated images and loss plots during or after training.
    *   Print training progress to the console.

## Configuration (Optional)

Training parameters such as `num_epochs`, `batch_size`, `learning_rate`, `lambda_cycle`, and `lambda_identity` can be configured directly within `main.py` or by passing command-line arguments (if implemented).

## Results

During and after training, you can expect:

*   **Console Output**: Loss values for generators and discriminators per batch/epoch.
*   **Generated Images**: Visualizations of image translations (e.g., MNIST digits converted to SVHN-style house numbers and vice versa), including cycle-reconstructed images, to observe the model's performance.
*   **Loss Plots**: Graphical representations of training losses over iterations, indicating the convergence and stability of the GAN.

## Technologies Used

*   **Python 3.x**
*   **PyTorch**: Deep learning framework
*   **torchvision**: For datasets, transforms, and model architectures
*   **matplotlib**: For plotting and visualization
*   **tqdm**: For progress bars

## Future Improvements

*   **Command-line Arguments**: Implement `argparse` for easier configuration of training parameters.
*   **Model Checkpointing**: Add functionality to save and load model weights.
*   **Quantitative Metrics**: Integrate quantitative evaluation metrics (e.g., FID, LPIPS) for objective performance assessment.
*   **TensorBoard/Weights & Biases**: Use a logging tool for better experiment tracking.
*   **Different Datasets**: Extend the framework to support other image-to-image translation tasks.
