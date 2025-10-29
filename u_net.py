import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    """
    U-Net Generator for image colorization.

    Architecture:
    - Takes grayscale L channel (1 channel) as input
    - Outputs a*b* color channels (2 channels)
    - Uses encoder-decoder structure with skip connections
    - Skip connections help preserve fine details lost during downsampling

    The U-Net consists of:
    1. Encoder: Progressively downsamples and increases feature channels
    2. Bottleneck: Smallest spatial resolution, highest feature dimension
    3. Decoder: Progressively upsamples and decreases feature channels
    4. Skip connections: Concatenate encoder features with decoder features
    """

    def __init__(self, input_channels=1, output_channels=2):
        """
        Initialize U-Net Generator.

        Args:
            input_channels (int): Number of input channels (1 for grayscale L channel)
            output_channels (int): Number of output channels (2 for a*b* channels)
        """
        super(UNetGenerator, self).__init__()

        # Encoder (Downsampling path)
        # Each encoder block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
        # Followed by MaxPool to reduce spatial dimensions
        self.enc1 = self.conv_block(input_channels, 64)  # 256x256x1 -> 256x256x64
        self.enc2 = self.conv_block(64, 128)             # 128x128x64 -> 128x128x128
        self.enc3 = self.conv_block(128, 256)            # 64x64x128 -> 64x64x256
        self.enc4 = self.conv_block(256, 512)            # 32x32x256 -> 32x32x512

        # Bottleneck - deepest point of the network
        # Captures high-level semantic information
        self.bottleneck = self.conv_block(512, 1024)     # 16x16x512 -> 16x16x1024

        # Decoder (Upsampling path)
        # Each decoder block processes upsampled features + skip connection
        self.dec4 = self.upconv_block(1024, 512)         # 32x32x1024 -> 32x32x512
        self.dec3 = self.upconv_block(1024, 256)         # 64x64x1024 -> 64x64x256 (1024 = 512 + 512 from skip)
        self.dec2 = self.upconv_block(512, 128)          # 128x128x512 -> 128x128x128 (512 = 256 + 256 from skip)
        self.dec1 = self.upconv_block(256, 64)           # 256x256x256 -> 256x256x64 (256 = 128 + 128 from skip)

        # Final output layer - maps features to color channels
        # 1x1 convolution to reduce from 128 channels (64 + 64 from skip) to 2 channels (a*b*)
        self.final = nn.Conv2d(128, output_channels, kernel_size=1)

        # Pooling and upsampling operations
        self.pool = nn.MaxPool2d(2)  # Reduces spatial dimensions by factor of 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Increases spatial dimensions by factor of 2

    def conv_block(self, in_channels, out_channels):
        """
        Standard convolutional block used in encoder and decoder.

        Each block consists of:
        1. 3x3 Convolution with padding=1 (preserves spatial dimensions)
        2. Batch Normalization (stabilizes training)
        3. ReLU activation (non-linearity)
        4. Another 3x3 Convolution with padding=1
        5. Batch Normalization
        6. ReLU activation

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Returns:
            nn.Sequential: The convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """
        Upsampling convolutional block used in decoder.

        Similar to conv_block but designed for processing upsampled features
        that will be concatenated with skip connections.

        Args:
            in_channels (int): Number of input channels (includes skip connection channels)
            out_channels (int): Number of output channels

        Returns:
            nn.Sequential: The upsampling convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through U-Net Generator.

        Process:
        1. Encoder: Extract features at multiple scales while downsampling
        2. Bottleneck: Process features at lowest resolution
        3. Decoder: Reconstruct image while upsampling, using skip connections
        4. Output: Generate a*b* color channels

        Args:
            x (torch.Tensor): Input grayscale L channel, shape (B, 1, H, W)

        Returns:
            torch.Tensor: Predicted a*b* channels, shape (B, 2, H, W), range [-1, 1]
        """
        # Encoder path - progressively downsample and extract features
        # Each level captures features at different scales
        enc1 = self.enc1(x)                    # 256x256x64 - fine details
        enc2 = self.enc2(self.pool(enc1))      # 128x128x128 - medium details
        enc3 = self.enc3(self.pool(enc2))      # 64x64x256 - larger structures
        enc4 = self.enc4(self.pool(enc3))      # 32x32x512 - high-level features

        # Bottleneck - deepest point with highest semantic understanding
        bottleneck = self.bottleneck(self.pool(enc4))  # 16x16x1024

        # Decoder path - progressively upsample and refine features
        # Skip connections preserve fine details lost during downsampling

        # First decoder level: combine bottleneck with enc4 features
        dec4 = self.dec4(self.upsample(bottleneck))    # 32x32x512
        dec4 = torch.cat([dec4, enc4], dim=1)          # 32x32x1024 (512+512)

        # Second decoder level: combine dec4 with enc3 features
        dec3 = self.dec3(self.upsample(dec4))          # 64x64x256
        dec3 = torch.cat([dec3, enc3], dim=1)          # 64x64x512 (256+256)

        # Third decoder level: combine dec3 with enc2 features
        dec2 = self.dec2(self.upsample(dec3))          # 128x128x128
        dec2 = torch.cat([dec2, enc2], dim=1)          # 128x128x256 (128+128)

        # Fourth decoder level: combine dec2 with enc1 features
        dec1 = self.dec1(self.upsample(dec2))          # 256x256x64
        dec1 = torch.cat([dec1, enc1], dim=1)          # 256x256x128 (64+64)

        # Final output layer - map features to color channels
        output = self.final(dec1)                      # 256x256x2
        return torch.tanh(output)  # Tanh ensures output is in range [-1, 1] for a*b* channels


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for adversarial training.

    Architecture:
    - Takes full LAB images (L + a*b* = 3 channels) as input
    - Outputs a probability map indicating real/fake patches
    - Uses strided convolutions for downsampling instead of pooling
    - LeakyReLU activations prevent gradient saturation

    The discriminator helps the generator produce realistic colors
    by learning to distinguish between real and generated color images.
    """

    def __init__(self, input_channels=3):
        """
        Initialize Discriminator.

        Args:
            input_channels (int): Number of input channels (3 for L+a*b*)
        """
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalize=True):
            """
            Single discriminator block with convolution, normalization, and activation.

            Args:
                in_channels (int): Number of input channels
                out_channels (int): Number of output channels
                normalize (bool): Whether to apply batch normalization

            Returns:
                list: List of layers for this block
            """
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Build discriminator network
        # Each block halves spatial dimensions while doubling channels
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),  # 256x256x3 -> 128x128x64
            *discriminator_block(64, 128),                              # 128x128x64 -> 64x64x128
            *discriminator_block(128, 256),                             # 64x64x128 -> 32x32x256
            *discriminator_block(256, 512),                             # 32x32x256 -> 16x16x512
            nn.Conv2d(512, 1, 4, padding=1),                          # 16x16x512 -> 15x15x1
            nn.Sigmoid()                                                # Output probabilities [0,1]
        )

    def forward(self, img):
        """
        Forward pass through discriminator.

        Args:
            img (torch.Tensor): Input LAB image, shape (B, 3, H, W)

        Returns:
            torch.Tensor: Probability map of real/fake patches, shape (B, 1, H', W')
        """
        return self.model(img)


class ColorGAN:
    """
    Complete GAN system for image colorization.

    Combines U-Net generator and PatchGAN discriminator for adversarial training.
    Uses both adversarial loss (for realism) and L1 reconstruction loss (for accuracy).

    Training Strategy:
    1. Generator tries to produce realistic colors that fool the discriminator
    2. Discriminator tries to distinguish real vs generated color images
    3. L1 loss ensures generated colors match ground truth
    4. Balance between adversarial and reconstruction losses creates high-quality colorization
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize ColorGAN with generator, discriminator, and optimizers.

        Args:
            device (str): Device to run training on ('cuda' or 'cpu')
        """
        self.device = device
        self.generator = UNetGenerator().to(device)
        self.discriminator = Discriminator().to(device)

        # Loss functions
        self.adversarial_loss = nn.BCELoss()  # Binary cross-entropy for real/fake classification
        self.l1_loss = nn.L1Loss()            # L1 loss for pixel-wise color accuracy

        # Optimizers - Adam with specific hyperparameters for GAN training
        # Lower learning rate and beta1=0.5 are common for stable GAN training
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Loss weights - balance between adversarial and reconstruction objectives
        self.lambda_l1 = 40  # High weight ensures color accuracy, typical for pix2pix-style models

    def train_step(self, real_L, real_ab):
        """
        Perform one training step of the GAN.

        This implements the standard GAN training procedure:
        1. Train generator to fool discriminator while minimizing reconstruction error
        2. Train discriminator to distinguish real from fake images

        Args:
            real_L (torch.Tensor): Real grayscale L channels, shape (B, 1, H, W)
            real_ab (torch.Tensor): Real color a*b* channels, shape (B, 2, H, W)

        Returns:
            dict: Dictionary containing all loss values for monitoring training progress
        """
        batch_size = real_L.size(0)

        # Create labels for adversarial training
        # Real labels = 1 (discriminator should output 1 for real images)
        # Fake labels = 0 (discriminator should output 0 for fake images)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # ============= TRAIN GENERATOR =============
        # Goal: Generate realistic colors that fool the discriminator
        # while maintaining color accuracy with L1 loss
        self.optimizer_G.zero_grad()

        # Generate fake color channels from grayscale input
        fake_ab = self.generator(real_L)  # Shape: (B, 2, H, W)

        # Combine L channel with both real and fake ab channels for discrimination
        fake_lab = torch.cat([real_L, fake_ab], dim=1)  # Shape: (B, 3, H, W)
        real_lab = torch.cat([real_L, real_ab], dim=1)  # Shape: (B, 3, H, W)

        # Adversarial loss: Generator wants discriminator to think fake images are real
        pred_fake = self.discriminator(fake_lab)  # Discriminator's prediction on fake images
        # Flatten spatial dimensions and average to get single probability per image
        pred_fake_flat = pred_fake.view(batch_size, -1).mean(dim=1, keepdim=True)
        loss_G_GAN = self.adversarial_loss(pred_fake_flat, real_labels)  # Want discriminator to output 1

        # L1 reconstruction loss: Ensures generated colors are close to ground truth
        # This is crucial for colorization - without it, colors could be realistic but wrong
        loss_G_L1 = self.l1_loss(fake_ab, real_ab)

        # Total generator loss: Balance between fooling discriminator and color accuracy
        # Lambda_l1 is high (100) because color accuracy is very important for colorization
        loss_G = loss_G_GAN + self.lambda_l1 * loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()

        # ============= TRAIN DISCRIMINATOR =============
        # Goal: Learn to distinguish real color images from generator's fake ones
        # This provides feedback to help generator improve
        self.optimizer_D.zero_grad()

        # Train on real images - discriminator should output 1 (real)
        pred_real = self.discriminator(real_lab)
        pred_real_flat = pred_real.view(batch_size, -1).mean(dim=1, keepdim=True)
        loss_D_real = self.adversarial_loss(pred_real_flat, real_labels)

        # Train on fake images - discriminator should output 0 (fake)
        # Detach fake_lab to prevent gradients flowing back to generator
        fake_lab_detached = fake_lab.detach()
        pred_fake = self.discriminator(fake_lab_detached)
        pred_fake_flat = pred_fake.view(batch_size, -1).mean(dim=1, keepdim=True)
        loss_D_fake = self.adversarial_loss(pred_fake_flat, fake_labels)

        # Total discriminator loss: Average of real and fake losses
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        self.optimizer_D.step()

        # Return all loss components for monitoring training progress
        return {
            'loss_G': loss_G.item(),           # Total generator loss
            'loss_G_GAN': loss_G_GAN.item(),   # Generator adversarial loss
            'loss_G_L1': loss_G_L1.item(),     # Generator reconstruction loss
            'loss_D': loss_D.item()            # Discriminator loss
        }

    def save_models(self, path_prefix):
        """
        Save both generator and discriminator model weights.

        Args:
            path_prefix (str): Path prefix for saving models.
                             Will create {path_prefix}_generator.pth and {path_prefix}_discriminator.pth
        """
        torch.save(self.generator.state_dict(), f"{path_prefix}_generator.pth")
        torch.save(self.discriminator.state_dict(), f"{path_prefix}_discriminator.pth")

    def load_models(self, path_prefix):
        """
        Load both generator and discriminator model weights.

        Args:
            path_prefix (str): Path prefix for loading models.
                             Should match the prefix used when saving
        """
        self.generator.load_state_dict(torch.load(f"{path_prefix}_generator.pth"))
        self.discriminator.load_state_dict(torch.load(f"{path_prefix}_discriminator.pth"))