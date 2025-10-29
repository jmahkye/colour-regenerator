import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from skimage import color
import glob


class ColorDataset(Dataset):
    """
    PyTorch Dataset for image colorization training with recursive directory search.
    """

    def __init__(self, image_dir, image_size=256, recursive=True, max_depth=None):
        """
        Initialize the ColorDataset.

        Args:
            image_dir (str): Directory containing training images
            image_size (int): Target size for resizing images (images will be square)
            recursive (bool): If True, search subdirectories recursively
            max_depth (int, optional): Maximum depth to search (None = unlimited)
                - 0: only immediate directory
                - 1: immediate directory + 1 level of subdirectories
                - 2: immediate directory + 2 levels, etc.
        """
        self.image_dir = image_dir
        self.image_size = image_size

        # Search for image files with common extensions
        self.image_paths = []
        extensions = ['JPEG', 'jpeg', 'jpg', 'JPG', 'png', 'PNG']

        if recursive and max_depth is None:
            # Search recursively with NO depth limit (goes infinitely deep)
            for ext in extensions:
                pattern = os.path.join(image_dir, f"**/*.{ext}")
                self.image_paths.extend(glob.glob(pattern, recursive=True))

            print(f"Found {len(self.image_paths)} images in {image_dir}")
            print("  (searched recursively in ALL subdirectories - unlimited depth)")

        elif recursive and max_depth is not None:
            # Search with depth limit
            for ext in extensions:
                self.image_paths.extend(self._find_images_with_depth(image_dir, ext, max_depth))

            print(f"Found {len(self.image_paths)} images in {image_dir}")
            print(f"  (searched recursively up to {max_depth} level(s) deep)")

        else:
            # Search only in the immediate directory (original behavior)
            for ext in extensions:
                pattern = os.path.join(image_dir, f"*.{ext}")
                self.image_paths.extend(glob.glob(pattern))

            print(f"Found {len(self.image_paths)} images in {image_dir}")
            print("  (searched only in immediate directory)")

        # Remove duplicates and sort for reproducibility
        self.image_paths = sorted(list(set(self.image_paths)))

        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to square image
            transforms.ToTensor()  # Convert PIL Image to tensor [0,1]
        ])

    def _find_images_with_depth(self, directory, extension, max_depth, current_depth=0):
        """
        Helper function to find images up to a specified depth.

        Args:
            directory (str): Directory to search
            extension (str): File extension to search for
            max_depth (int): Maximum depth to search
            current_depth (int): Current depth level

        Returns:
            list: List of image paths found
        """
        image_paths = []

        # Find images at current level
        pattern = os.path.join(directory, f"*.{extension}")
        image_paths.extend(glob.glob(pattern))

        # If we haven't reached max depth, search subdirectories
        if current_depth < max_depth:
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isdir(item_path):
                        # Recursively search subdirectory
                        image_paths.extend(
                            self._find_images_with_depth(
                                item_path, extension, max_depth, current_depth + 1
                            )
                        )
            except PermissionError:
                pass  # Skip directories we can't access

        return image_paths

        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and process a single image for training.

        Args:
            idx (int): Index of the image to load

        Returns:
            tuple: (L_tensor, ab_tensor)
                - L_tensor: Grayscale L channel, shape (1, H, W), range [-1, 1]
                - ab_tensor: Color a*b* channels, shape (2, H, W), range [-1, 1]
        """
        img_path = self.image_paths[idx]

        try:
            # Load RGB image and ensure it's in RGB format
            rgb_img = Image.open(img_path).convert('RGB')

            # Apply transforms: resize and convert to tensor
            rgb_img = self.transform(rgb_img)  # Shape: (3, H, W), range [0, 1]

            # Convert tensor to numpy for color space conversion
            rgb_np = rgb_img.permute(1, 2, 0).numpy()

            # Convert from RGB to LAB color space
            lab_img = color.rgb2lab(rgb_np)

            # Extract and normalize LAB channels
            # L channel: [0, 100] → [-1, 1]
            L = lab_img[:, :, 0] / 50.0 - 1.0

            # a*b* channels: [-128, 127] → [-1, 1]
            ab = lab_img[:, :, 1:] / 128.0

            # Convert to PyTorch tensors
            L_tensor = torch.FloatTensor(L).unsqueeze(0)  # (1, H, W)
            ab_tensor = torch.FloatTensor(ab).permute(2, 0, 1)  # (2, H, W)

            return L_tensor, ab_tensor

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback: return random tensors if image loading fails
            L_tensor = torch.randn(1, self.image_size, self.image_size)
            ab_tensor = torch.randn(2, self.image_size, self.image_size)
            return L_tensor, ab_tensor


def create_data_loader(image_dir, batch_size=16, image_size=256, shuffle=True,
                       num_workers=4, recursive=True, max_depth=None):
    """
    Create a PyTorch DataLoader for training the colorization model.

    Args:
        image_dir (str): Directory containing training images
        batch_size (int): Number of images per training batch
        image_size (int): Target size for image resizing
        shuffle (bool): Whether to shuffle the dataset each epoch
        num_workers (int): Number of parallel workers for data loading
        recursive (bool): If True, search subdirectories recursively
        max_depth (int, optional): Maximum depth to search (None = unlimited)

    Returns:
        DataLoader: PyTorch DataLoader ready for training
    """
    dataset = ColorDataset(image_dir, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,          # Reduced to save RAM
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # ADD THIS - reuses workers
    )
    return dataloader


def lab_to_rgb_tensor(L, ab):
    """
    Convert LAB tensors back to RGB format for visualization and saving.

    Args:
        L (torch.Tensor): L channel tensors, shape (B, 1, H, W), range [-1, 1]
        ab (torch.Tensor): a*b* channel tensors, shape (B, 2, H, W), range [-1, 1]

    Returns:
        torch.Tensor: RGB images, shape (B, 3, H, W), range [0, 1]
    """
    # Denormalize LAB values
    L = (L + 1.0) * 50.0  # [-1, 1] → [0, 100]
    ab = ab * 128.0  # [-1, 1] → [-128, 127]

    # Combine L and ab channels
    lab = torch.cat([L, ab], dim=1)  # Shape: (B, 3, H, W)

    # Convert each image in the batch from LAB to RGB
    rgb_images = []
    for i in range(lab.size(0)):
        lab_np = lab[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        rgb_np = color.lab2rgb(lab_np)  # [0, 1]
        rgb_tensor = torch.FloatTensor(rgb_np).permute(2, 0, 1)  # (3, H, W)
        rgb_images.append(rgb_tensor)

    return torch.stack(rgb_images)  # Shape: (B, 3, H, W)
