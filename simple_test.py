import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
import argparse
import os

from u_net import UNetGenerator
from dataset import lab_to_rgb_tensor

class ColorPredictor:
    """
    Inference system for the trained image colorization model.

    This class provides a complete inference pipeline for colorizing images:
    - Load trained model weights
    - Process input images (resize, convert to LAB)
    - Generate colorizations using the trained generator
    - Convert results back to RGB for visualization/saving
    - Handle batch processing and single image inference

    Key Features:
    - Flexible input handling (file paths, PIL Images, directories)
    - Automatic image preprocessing and postprocessing
    - Support for different output formats and sizes
    - Batch processing for efficiency
    - Comparison utilities for evaluation

    The predictor handles the full pipeline from raw RGB images to colorized outputs,
    taking care of all the color space conversions and normalization needed.
    """

    def __init__(self, model_path, device=None):
        """
        Initialize the ColorPredictor with a trained model.

        Args:
            model_path (str): Path to the saved model (without _generator.pth suffix)
            device (torch.device, optional): Device to run inference on. Auto-detects if None.
        """
        # Set up computation device (GPU preferred for speed)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load the trained generator model
        self.generator = UNetGenerator().to(self.device)
        self.generator.load_state_dict(torch.load(f"{model_path}_generator.pth", map_location=self.device))
        self.generator.eval()  # Set to evaluation mode (no dropout, fixed batchnorm)
        print(f"Loaded model from {model_path}")

    def preprocess_image(self, image_path, target_size=256):
        """
        Load and preprocess an image for colorization inference.

        This method handles the complete preprocessing pipeline:
        1. Load image from file or accept PIL Image object
        2. Resize to model's expected input size
        3. Convert RGB to LAB color space
        4. Extract and normalize L channel for the model
        5. Convert to PyTorch tensor format

        Args:
            image_path (str or PIL.Image): Path to image file or PIL Image object
            target_size (int): Size to resize image to (square)

        Returns:
            tuple: (L_tensor, original_size)
                - L_tensor: Preprocessed L channel, shape (1, 1, H, W), range [-1, 1]
                - original_size: Original image dimensions for final resizing
        """
        # Load image - handle both file paths and PIL Image objects
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path

        # Store original size for later restoration
        original_size = img.size

        # Resize to model's expected input size using high-quality resampling
        img = img.resize((target_size, target_size), Image.LANCZOS)

        # Convert PIL Image to numpy array and normalize to [0, 1]
        img_np = np.array(img) / 255.0

        # Convert from RGB to LAB color space
        # This separates lightness (L) from color information (a*, b*)
        lab_img = color.rgb2lab(img_np)

        # Extract L channel and normalize for neural network input
        # L channel range: [0, 100] → [-1, 1] (standard neural network range)
        L = lab_img[:, :, 0] / 50.0 - 1.0

        # Convert to PyTorch tensor with correct dimensions
        # Add batch dimension and channel dimension: (H, W) → (1, 1, H, W)
        L_tensor = torch.FloatTensor(L).unsqueeze(0).unsqueeze(0)

        return L_tensor, original_size

    def colorize_image(self, image_path, output_path=None, target_size=256):
        """
        Colorize a single image using the trained model.

        Complete inference pipeline:
        1. Preprocess input image (resize, convert to LAB, normalize)
        2. Run inference using the trained generator
        3. Convert model output back to RGB
        4. Resize to original dimensions
        5. Save result if requested

        Args:
            image_path (str or PIL.Image): Input image path or PIL Image object
            output_path (str, optional): Path to save colorized image. If None, don't save.
            target_size (int): Size for model inference (images are resized back afterwards)

        Returns:
            PIL.Image: Colorized image as PIL Image object
        """
        # Preprocess the input image for model inference
        L_tensor, original_size = self.preprocess_image(image_path, target_size)
        L_tensor = L_tensor.to(self.device)

        # Generate colorization using the trained model
        with torch.no_grad():  # Disable gradients for faster inference
            fake_ab = self.generator(L_tensor)  # Generate a*b* color channels

        # Convert model output back to RGB format for visualization
        rgb_tensor = lab_to_rgb_tensor(L_tensor, fake_ab)
        rgb_img = rgb_tensor[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        # Ensure values are in valid range [0, 1]
        rgb_img = np.clip(rgb_img, 0, 1)

        # Convert to PIL Image with 8-bit values and resize to original dimensions
        result_img = Image.fromarray((rgb_img * 255).astype(np.uint8))
        result_img = result_img.resize(original_size, Image.LANCZOS)

        # Save the result if output path is provided
        if output_path:
            result_img.save(output_path)
            print(f"Colorized image saved to {output_path}")

        return result_img

    def colorize_batch(self, image_dir, output_dir, max_images=None):
        """
        Colorize all images in a directory for batch processing.

        This method provides efficient batch processing capabilities:
        - Automatically finds all image files in the input directory
        - Processes each image through the colorization pipeline
        - Saves results with organized naming convention
        - Provides progress tracking and error handling
        - Supports limiting the number of images processed

        Args:
            image_dir (str): Directory containing input images
            output_dir (str): Directory to save colorized results
            max_images (int, optional): Maximum number of images to process. None for all.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Find all image files with common extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])

        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]

        print(f"Found {len(image_files)} images to colorize")

        # Process each image with progress tracking
        for i, filename in enumerate(image_files):
            input_path = os.path.join(image_dir, filename)
            output_filename = f"colorized_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            try:
                # Colorize and save the image
                self.colorize_image(input_path, output_path)
                print(f"Processed {i+1}/{len(image_files)}: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    def compare_images(self, image_path, output_path=None):
        """
        Create a side-by-side comparison image for evaluation.

        This method generates a comprehensive comparison showing:
        1. Grayscale input (what the model sees)
        2. Model's colorization output
        3. Original color image (ground truth)

        This comparison is essential for:
        - Visual evaluation of model performance
        - Qualitative assessment of colorization quality
        - Identifying strengths and weaknesses of the model
        - Creating presentation materials and documentation

        Args:
            image_path (str): Path to the input color image
            output_path (str, optional): Path to save comparison image

        Returns:
            PIL.Image: Comparison image with three panels side by side
        """
        # Load the original color image
        original_img = Image.open(image_path).convert('RGB')

        # Create grayscale version to show model input
        # Convert to L (grayscale) then back to RGB for consistent format
        grayscale_img = original_img.convert('L').convert('RGB')

        # Generate colorization using the trained model
        colorized_img = self.colorize_image(image_path)

        # Create side-by-side comparison image
        width, height = original_img.size
        comparison = Image.new('RGB', (width * 3, height))

        # Arrange images: grayscale | colorized | original
        comparison.paste(grayscale_img, (0, 0))           # Left: input
        comparison.paste(colorized_img, (width, 0))       # Middle: model output
        comparison.paste(original_img, (width * 2, 0))    # Right: ground truth

        # Save comparison if output path is provided
        if output_path:
            comparison.save(output_path)
            print(f"Comparison saved to {output_path}")

        return comparison

def main():
    """
    Command-line interface for the image colorization inference system.

    This function provides a flexible CLI for various colorization tasks:
    - Single image colorization
    - Batch processing of image directories
    - Side-by-side comparison generation
    - Flexible input/output handling

    Example Usage:
        # Colorize a single image
        python inference.py --model checkpoints/final_model --input photo.jpg

        # Create comparison (grayscale | colorized | original)
        python inference.py --model checkpoints/final_model --input photo.jpg --compare

        # Batch process a directory
        python inference.py --model checkpoints/final_model --input images/ --batch

        # Limit batch processing to first 10 images
        python inference.py --model checkpoints/final_model --input images/ --batch --max_images 10
    """
    parser = argparse.ArgumentParser(
        description='Colorize black and white images using trained GAN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:     python inference.py --model final_model --input photo.jpg
  Comparison:       python inference.py --model final_model --input photo.jpg --compare
  Batch process:    python inference.py --model final_model --input images/ --batch
  Limited batch:    python inference.py --model final_model --input images/ --batch --max_images 10
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (without _generator.pth suffix)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--output', type=str,
                       help='Output path or directory (auto-generated if not specified)')
    parser.add_argument('--compare', action='store_true',
                       help='Create side-by-side comparison (grayscale | colorized | original)')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of images')
    parser.add_argument('--max_images', type=int,
                       help='Maximum number of images to process in batch mode')

    args = parser.parse_args()

    # Initialize the colorization predictor with the trained model
    predictor = ColorPredictor(args.model)

    if args.batch:
        # Batch processing mode: process all images in a directory
        output_dir = args.output or f"{args.input}_colorized"
        predictor.colorize_batch(args.input, output_dir, args.max_images)
    else:
        # Single image processing mode
        if args.compare:
            # Create side-by-side comparison image
            output_path = args.output or f"{args.input}_comparison.jpg"
            predictor.compare_images(args.input, output_path)
        else:
            # Simple colorization
            output_path = args.output or f"{args.input}_colorized.jpg"
            predictor.colorize_image(args.input, output_path)

if __name__ == "__main__":
    main()