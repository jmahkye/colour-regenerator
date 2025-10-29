import torch
import os
import time
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

from u_net import ColorGAN
from dataset import create_data_loader, lab_to_rgb_tensor

class Trainer:
    """
    Complete training system for the image colorization GAN.

    This class handles all aspects of GAN training:
    - Data loading and batching
    - Training loop with loss tracking
    - Model checkpointing and resuming
    - Sample generation for monitoring progress
    - Loss visualization and logging

    The trainer implements best practices for GAN training:
    - Regular checkpointing to prevent loss of progress
    - Sample image generation to visually monitor quality
    - Loss tracking for debugging and optimization
    - Robust error handling and progress reporting
    """

    def __init__(self, data_dir, batch_size=16, learning_rate=0.0002, image_size=256):
        """
        Initialize the training system.

        Sets up the GAN model, data loading, output directories, and training state.
        Automatically detects available hardware (GPU vs CPU) for optimal training speed.

        Args:
            data_dir (str): Directory containing training images
            batch_size (int): Number of images per training batch
            learning_rate (float): Learning rate for optimizers (unused in current implementation)
            image_size (int): Size to resize images to (square images)
        """
        # Hardware detection - use GPU if available for faster training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize the GAN model (generator + discriminator + optimizers)
        self.gan = ColorGAN(device=self.device)

        # Set up data loading pipeline
        self.dataloader = create_data_loader(
            data_dir,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True  # Shuffle for better training
        )
        print(f"Dataset size: {len(self.dataloader.dataset)}")

        # Store training parameters
        self.batch_size = batch_size
        self.image_size = image_size

        # Create output directories for saving training artifacts
        os.makedirs('checkpoints', exist_ok=True)  # Model weights
        os.makedirs('samples', exist_ok=True)      # Sample images
        os.makedirs('logs', exist_ok=True)         # Training logs and plots

        # Initialize training history for loss tracking and visualization
        self.history = {
            'loss_G': [],        # Total generator loss
            'loss_G_GAN': [],    # Generator adversarial loss
            'loss_G_L1': [],     # Generator reconstruction loss
            'loss_D': []         # Discriminator loss
        }

    def save_sample_images(self, epoch, num_samples=4):
        """
        Generate and save sample colorized images for visual progress monitoring.

        This function creates side-by-side comparisons showing:
        1. Original grayscale input (L channel only)
        2. Model's colorization output
        3. Ground truth color image

        This visual feedback is crucial for:
        - Monitoring training progress
        - Detecting mode collapse or other training issues
        - Evaluating colorization quality subjectively
        - Debugging model performance

        Args:
            epoch (int): Current training epoch (used in filename)
            num_samples (int): Number of sample images to generate
        """
        # Set generator to evaluation mode (disables dropout, batchnorm updates)
        self.gan.generator.eval()

        with torch.no_grad():  # Disable gradient computation for efficiency
            # Get a fresh batch of real data
            data_iter = iter(self.dataloader)
            real_L, real_ab = next(data_iter)

            # Use only the first few samples to keep output manageable
            real_L = real_L[:num_samples].to(self.device)
            real_ab = real_ab[:num_samples].to(self.device)

            # Generate colorization using the current model
            fake_ab = self.gan.generator(real_L)

            # Convert all tensor formats back to RGB for visualization
            real_rgb = lab_to_rgb_tensor(real_L, real_ab)        # Ground truth colors
            fake_rgb = lab_to_rgb_tensor(real_L, fake_ab)        # Model's colorization

            # Create grayscale version by setting color channels to zero
            gray_rgb = lab_to_rgb_tensor(real_L, torch.zeros_like(real_ab))

            # Create comparison grid: [grayscale | model output | ground truth]
            # This layout makes it easy to see input, prediction, and target
            comparison = torch.cat([gray_rgb, fake_rgb, real_rgb], dim=0)

            # Save the comparison image with epoch number in filename
            save_image(
                comparison,
                f'samples/epoch_{epoch:04d}.png',
                nrow=num_samples,      # Number of images per row
                normalize=True,        # Normalize to [0,1] range
                value_range=(0, 1)     # Expected input range
            )

        # Return generator to training mode
        self.gan.generator.train()

    def plot_losses(self):
        """
        Generate and save training loss plots for monitoring and analysis.

        Creates comprehensive loss visualizations showing:
        1. Generator losses (total, adversarial, and L1 reconstruction)
        2. Discriminator loss

        These plots are essential for:
        - Monitoring training stability
        - Detecting convergence or divergence
        - Balancing generator vs discriminator training
        - Debugging training issues (mode collapse, vanishing gradients)

        The plots help identify:
        - If losses are decreasing over time
        - If generator and discriminator are balanced
        - If L1 loss is dominating (ensuring color accuracy)
        - Training stability and convergence patterns
        """
        # Create epoch range for x-axis
        epochs = range(1, len(self.history['loss_G']) + 1)

        # Create a figure with two subplots side by side
        plt.figure(figsize=(12, 4))

        # Left subplot: Generator losses (multiple components)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['loss_G'], label='Generator Total', linewidth=2)
        plt.plot(epochs, self.history['loss_G_GAN'], label='Generator Adversarial', alpha=0.8)
        plt.plot(epochs, self.history['loss_G_L1'], label='Generator L1 (Reconstruction)', alpha=0.8)
        plt.title('Generator Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Right subplot: Discriminator loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['loss_D'], label='Discriminator', color='red', linewidth=2)
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot with high quality
        plt.tight_layout()
        plt.savefig('logs/training_losses.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory

    def train(self, num_epochs=100, save_every=10, sample_every=5):
        """
        Main training loop for the GAN model.

        Implements the complete training procedure including:
        - Epoch-by-epoch training with batch processing
        - Loss tracking and averaging
        - Regular checkpointing and sample generation
        - Progress monitoring and time estimation
        - Robust error handling and recovery

        Training Strategy:
        1. For each epoch, iterate through all training batches
        2. Train both generator and discriminator on each batch
        3. Track and average losses across the epoch
        4. Periodically save models and generate sample images
        5. Create loss visualizations for monitoring

        Args:
            num_epochs (int): Total number of training epochs
            save_every (int): Save model checkpoints every N epochs
            sample_every (int): Generate sample images every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # Initialize loss tracking for this epoch
            epoch_losses = {
                'loss_G': [],        # Generator total loss
                'loss_G_GAN': [],    # Generator adversarial loss
                'loss_G_L1': [],     # Generator reconstruction loss
                'loss_D': []         # Discriminator loss
            }

            # Ensure models are in training mode (enables dropout, batchnorm updates)
            self.gan.generator.train()
            self.gan.discriminator.train()

            # Process all batches in the current epoch
            for batch_idx, (real_L, real_ab) in enumerate(self.dataloader):
                # Move data to GPU if available
                real_L = real_L.to(self.device)
                real_ab = real_ab.to(self.device)

                # Perform one training step (update both generator and discriminator)
                losses = self.gan.train_step(real_L, real_ab)

                # Accumulate losses for epoch averaging
                for key in epoch_losses:
                    epoch_losses[key].append(losses[key])

                # Print progress periodically to monitor training
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}/{num_epochs} | Batch {batch_idx}/{len(self.dataloader)} | "
                          f"G: {losses['loss_G']:.4f} | D: {losses['loss_D']:.4f} | "
                          f"L1: {losses['loss_G_L1']:.4f}")

            # Calculate average losses for the epoch
            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

            # Store epoch averages in training history
            for key in self.history:
                self.history[key].append(avg_losses[key])

            # Print comprehensive epoch summary
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs} completed in {elapsed/3600:.2f}h")
            print(f"Avg Losses - G: {avg_losses['loss_G']:.4f} | "
                  f"D: {avg_losses['loss_D']:.4f} | L1: {avg_losses['loss_G_L1']:.4f}")
            print("-" * 80)

            # Generate and save sample images for visual progress monitoring
            if epoch % sample_every == 0:
                self.save_sample_images(epoch)

            # Save model checkpoints and update loss plots
            if epoch % save_every == 0:
                self.gan.save_models(f'checkpoints/epoch_{epoch:04d}')
                self.plot_losses()
                print(f"Saved checkpoint at epoch {epoch}")

        # Final save after training completion
        self.gan.save_models('checkpoints/final_model')
        self.plot_losses()
        print(f"\nTraining completed! Total time: {(time.time() - start_time)/3600:.2f}h")

    def resume_training(self, checkpoint_path, num_epochs=100):
        """
        Resume training from a previously saved checkpoint.

        This function allows you to continue training from where you left off,
        which is essential for long training runs that may be interrupted.

        Args:
            checkpoint_path (str): Path to the checkpoint (without _generator.pth suffix)
            num_epochs (int): Number of additional epochs to train
        """
        self.gan.load_models(checkpoint_path)
        print(f"Resumed training from {checkpoint_path}")
        self.train(num_epochs)

if __name__ == "__main__":
    """
    Main training script execution.

    This section runs when the script is executed directly (not imported).
    It sets up the training configuration and starts the training process.

    Configuration Notes:
    - BATCH_SIZE: Adjust based on your GPU memory (8 for 8GB, 16+ for 16GB+)
    - NUM_EPOCHS: 100 is usually sufficient for good results
    - IMAGE_SIZE: 256x256 is a good balance between quality and training speed

    Expected Training Time:
    - GPU (RTX 3080): ~6-8 hours for 100 epochs with 10K images
    - GPU (RTX 4090): ~3-4 hours for 100 epochs with 10K images
    - CPU: Not recommended (would take days)
    """
    # Training configuration parameters
    DATA_DIR = "data/coco"  # Directory containing training images
    BATCH_SIZE = 4                          # Adjust based on GPU memory (8=safe, 16+=high-end)
    NUM_EPOCHS = 10                        # Number of training epochs
    IMAGE_SIZE = 256                        # Input image size (256x256 pixels)

    # Initialize the training system
    trainer = Trainer(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    # Start the training process
    # This will create checkpoints, samples, and logs automatically
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_every=2,      # Save model every 10 epochs
        sample_every=5      # Generate sample images every 5 epochs
    )