"""
LAB Color Space Demonstration Script

This script demonstrates the basic LAB color space conversion that forms the foundation
of our image colorization approach. It shows how to:

1. Load a color image
2. Convert from RGB to LAB color space
3. Split LAB into separate channels
4. Understand the input/output structure for colorization

LAB Color Space Explanation:
- L channel: Lightness/brightness information (0-100)
- a* channel: Green-Red color axis (-128 to +127)
- b* channel: Blue-Yellow color axis (-128 to +127)

Why LAB for Colorization:
1. Separates brightness from color information
2. More perceptually uniform than RGB
3. L channel represents what humans see as grayscale
4. a*b* channels contain only color information
5. Easier for neural networks to learn color mappings

This separation is perfect for colorization:
- Input: L channel (grayscale)
- Output: a*b* channels (color)
- Combine: Full LAB → convert back to RGB
"""

from skimage import color, io

# Load a color image from the training dataset
# This demonstrates the same processing pipeline used in training
rgb_image = io.imread('data/imagenet/test/images/test_0.jpeg')

print("=== LAB Color Space Conversion Demo ===")
print(f"Original RGB image shape: {rgb_image.shape}")
print(f"RGB value range: [{rgb_image.min()}, {rgb_image.max()}]")

# Convert the image from RGB to LAB color space
# This is the core transformation for colorization training
lab_image = color.rgb2lab(rgb_image)

print(f"\nLAB image shape: {lab_image.shape}")
print(f"LAB value ranges:")
print(f"  L channel: [{lab_image[:,:,0].min():.1f}, {lab_image[:,:,0].max():.1f}] (lightness)")
print(f"  a* channel: [{lab_image[:,:,1].min():.1f}, {lab_image[:,:,1].max():.1f}] (green-red)")
print(f"  b* channel: [{lab_image[:,:,2].min():.1f}, {lab_image[:,:,2].max():.1f}] (blue-yellow)")

# Split the LAB image into separate channels for training
L_channel = lab_image[:, :, 0]    # Lightness - this becomes our model INPUT
ab_channels = lab_image[:, :, 1:] # Color channels - this becomes our model TARGET

print(f"\n=== Training Data Structure ===")
print(f"Model INPUT (L channel) shape: {L_channel.shape}")
print(f"Model TARGET (a*b* channels) shape: {ab_channels.shape}")

print(f"\n=== Training Pipeline ===")
print("1. Load RGB image")
print("2. Convert RGB → LAB")
print("3. Extract L channel as input (grayscale)")
print("4. Extract a*b* channels as target (color)")
print("5. Train model: L → a*b*")
print("6. Inference: L + predicted_ab → LAB → RGB")

print(f"\n=== Normalization for Neural Networks ===")
print("For training, these values are normalized:")
print("  L channel: [0, 100] → [-1, 1] via (L/50.0 - 1.0)")
print("  a*b* channels: [-128, 127] → [-1, 1] via (ab/128.0)")
print("This puts all values in the standard neural network range of [-1, 1]")