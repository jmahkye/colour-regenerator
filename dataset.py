import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from skimage import color
from pathlib import Path
import glob

class ColorDataset(Dataset):
    """
    PyTorch Dataset for image colorization training.

    This dataset loads color images and converts them to LAB color space for training:
    - Input: L channel (grayscale/lightness)
    - Target: a*b* channels (color information)

    The LAB color space is ideal for colorization because:
    1. L channel contains all brightness information (like grayscale)
    2. a*b* channels contain only color information
    3. More perceptually uniform than RGB
    4. Easier to learn color mappings

    Processing Pipeline:
    1. Load RGB image from disk
    2. Resize to consistent size for training
    3. Convert RGB → LAB color space
    4. Split into L (input) and a*b* (target) channels
    5. Normalize values to [-1, 1] range for neural network training
    """

    def __init__(self, image_dir, image_size=256):
        """
        Initialize the ColorDataset.

        Args:
            image_dir (str): Directory containing training images
            image_size (int): Target size for resizing images (images will be square)
        """
        self.image_dir = image_dir
        self.image_size = image_size

        # Search recursively for image files with common extensions (case-insensitive)
        image_dir_path = Path(image_dir)
        self.image_paths = []

        # Search for common image formats
        for extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            # Case-insensitive search using rglob
            self.image_paths.extend([str(p) for p in image_dir_path.rglob(f'*.{extension}')])
            self.image_paths.extend([str(p) for p in image_dir_path.rglob(f'*.{extension.upper()}')])

        # Remove duplicates
        self.image_paths = list(set(self.image_paths))

        self.image_paths = self.image_paths[:20000]

        print(f"Found {len(self.image_paths)} images in {image_dir}")
        print(f"  (searched recursively in ALL subdirectories - unlimited depth)")

        if len(self.image_paths) == 0:
            print(f"WARNING: No images found! Check that {image_dir} contains image files.")
            print(f"Supported formats: .jpg, .jpeg, .png, .bmp, .tiff (case-insensitive)")

        # Image preprocessing transforms
        # Resize to consistent size and convert to tensor for PyTorch
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to square image
            transforms.ToTensor()  # Convert PIL Image to tensor [0,1]
        ])

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and process a single image for training.

        This method implements the core data processing pipeline:
        1. Load RGB image from disk
        2. Apply transforms (resize, normalize)
        3. Convert RGB to LAB color space
        4. Split into input (L) and target (a*b*) channels
        5. Normalize for neural network training

        Args:
            idx (int): Index of the image to load

        Returns:
            tuple: (L_tensor, ab_tensor)
                - L_tensor: Grayscale L channel, shape (1, H, W), range [-1, 1]
                - ab_tensor: Color a*b* channels, shape (2, H, W), range [-1, 1]
        """
        img_path = self.image_paths[idx]

        try:
            # Load RGB image and ensure it's in RGB format (not RGBA or grayscale)
            rgb_img = Image.open(img_path).convert('RGB')

            # Apply transforms: resize to target size and convert to tensor
            rgb_img = self.transform(rgb_img)  # Shape: (3, H, W), range [0, 1]

            # Convert tensor back to numpy for color space conversion
            # PyTorch tensors are (C, H, W), but skimage expects (H, W, C)
            rgb_np = rgb_img.permute(1, 2, 0).numpy()

            # Convert from RGB to LAB color space
            # LAB separates lightness (L) from color information (a*, b*)
            lab_img = color.rgb2lab(rgb_np)

            # Extract and normalize LAB channels for neural network training
            # L channel normalization: [0, 100] → [-1, 1]
            # This puts lightness in the standard range for neural networks
            L = lab_img[:, :, 0] / 50.0 - 1.0

            # A and B channel normalization: [-128, 127] → [-1, 1]
            # Color channels have different range, normalize to [-1, 1]
            ab = lab_img[:, :, 1:] / 128.0

            # Convert to PyTorch tensors with correct dimensions
            L_tensor = torch.FloatTensor(L).unsqueeze(0)  # Add channel dimension: (1, H, W)
            ab_tensor = torch.FloatTensor(ab).permute(2, 0, 1)  # Reorder to: (2, H, W)

            return L_tensor, ab_tensor

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback: return random tensors if image loading fails
            # This prevents training from crashing on corrupted images
            L_tensor = torch.randn(1, self.image_size, self.image_size)
            ab_tensor = torch.randn(2, self.image_size, self.image_size)
            return L_tensor, ab_tensor

def create_data_loader(image_dir, batch_size=16, image_size=256, shuffle=True, num_workers=4):
    """
    Create a PyTorch DataLoader for training the colorization model.

    This function wraps the ColorDataset in a DataLoader with optimized settings
    for training neural networks. The DataLoader handles:
    - Batching multiple images together
    - Shuffling data for better training
    - Parallel data loading for efficiency
    - Memory pinning for faster GPU transfer

    Args:
        image_dir (str): Directory containing training images
        batch_size (int): Number of images per training batch
        image_size (int): Target size for image resizing
        shuffle (bool): Whether to shuffle the dataset each epoch
        num_workers (int): Number of parallel workers for data loading

    Returns:
        DataLoader: PyTorch DataLoader ready for training
    """
    dataset = ColorDataset(image_dir, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,              # Shuffle data each epoch for better training
        num_workers=num_workers,      # Parallel data loading for speed
        pin_memory=True,              # Faster GPU transfer if using CUDA
        drop_last=True                # Drop incomplete last batch for consistent batch sizes
    )
    return dataloader

def lab_to_rgb_tensor(L, ab):
    """
    Convert LAB tensors back to RGB format for visualization and saving.

    This function reverses the LAB preprocessing done in the dataset:
    1. Denormalize LAB values from [-1, 1] back to original ranges
    2. Combine L and a*b* channels into full LAB images
    3. Convert LAB to RGB color space
    4. Return as PyTorch tensors ready for visualization

    This is essential for:
    - Visualizing training progress
    - Saving colorized results
    - Evaluating model performance

    Args:
        L (torch.Tensor): L channel tensors, shape (B, 1, H, W), range [-1, 1]
        ab (torch.Tensor): a*b* channel tensors, shape (B, 2, H, W), range [-1, 1]

    Returns:
        torch.Tensor: RGB images, shape (B, 3, H, W), range [0, 1]
    """
    # Denormalize LAB values back to their original ranges
    L = (L + 1.0) * 50.0  # [-1, 1] → [0, 100] (lightness range)
    ab = ab * 128.0       # [-1, 1] → [-128, 127] (color range)

    # Combine L and ab channels into full LAB images
    lab = torch.cat([L, ab], dim=1)  # Shape: (B, 3, H, W)

    # Convert each image in the batch from LAB to RGB
    rgb_images = []
    for i in range(lab.size(0)):
        # Extract single image and convert tensor format for skimage
        lab_np = lab[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        # Convert LAB to RGB using skimage
        rgb_np = color.lab2rgb(lab_np)  # Returns values in [0, 1]

        # Convert back to PyTorch tensor format
        rgb_tensor = torch.FloatTensor(rgb_np).permute(2, 0, 1)  # (3, H, W)
        rgb_images.append(rgb_tensor)

    # Stack all images back into a batch
    return torch.stack(rgb_images)  # Shape: (B, 3, H, W)