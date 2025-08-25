import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras import layers, Model
import warnings

warnings.filterwarnings('ignore')


class colourisationModel:
    def __init__(self, input_size=(256, 256, 1)):
        self.input_size = input_size
        self.model = None

    def build_model(self):
        """Build U-Net style architecture for colourisation"""
        inputs = layers.Input(shape=self.input_size)

        # Encoder (downsampling path)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        # Bottleneck
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

        # Decoder (upsampling path with skip connections)
        up5 = layers.UpSampling2D(size=(2, 2))(conv4)
        merge5 = layers.concatenate([up5, conv3], axis=3)
        conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
        conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

        up6 = layers.UpSampling2D(size=(2, 2))(conv5)
        merge6 = layers.concatenate([up6, conv2], axis=3)
        conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
        conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

        up7 = layers.UpSampling2D(size=(2, 2))(conv6)
        merge7 = layers.concatenate([up7, conv1], axis=3)
        conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
        conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

        # Output layer - 2 channels for a,b in LAB colour space
        outputs = layers.Conv2D(2, 1, activation='tanh', padding='same')(conv7)

        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model

    def compile_model(self):
        """Compile the model with optimizer and loss"""
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()


class DataProcessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def rgb_to_lab(self, rgb_image):
        """Convert RGB image to LAB colour space"""
        lab_image = cv2.cvtcolour(rgb_image, cv2.colour_RGB2LAB)
        return lab_image

    def lab_to_rgb(self, lab_image):
        """Convert LAB image back to RGB"""
        rgb_image = cv2.cvtcolour(lab_image.astype(np.uint8), cv2.colour_LAB2RGB)
        return rgb_image

    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, None

        # Convert BGR to RGB
        image = cv2.cvtcolour(image, cv2.colour_BGR2RGB)

        # Resize image
        image = cv2.resize(image, self.target_size)

        # Convert to LAB
        lab_image = self.rgb_to_lab(image)

        # Split LAB channels
        L = lab_image[:, :, 0]  # Lightness (grayscale input)
        ab = lab_image[:, :, 1:]  # colour channels (target output)

        # Normalize
        L = L.astype(np.float32) / 255.0  # Normalize L to [0,1]
        ab = ab.astype(np.float32) / 127.5 - 1.0  # Normalize ab to [-1,1]

        return L, ab

    def prepare_dataset(self, image_paths):
        """Prepare dataset from list of image paths"""
        X, y = [], []

        print(f"Processing {len(image_paths)} images...")

        for i, path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Processed {i}/{len(image_paths)} images")

            L, ab = self.preprocess_image(path)
            if L is not None:
                X.append(L[..., np.newaxis])  # Add channel dimension
                y.append(ab)

        return np.array(X), np.array(y)

    def postprocess_prediction(self, L_input, ab_pred):
        """Convert model prediction back to RGB image"""
        # Denormalize
        L = (L_input.squeeze() * 255.0).astype(np.uint8)
        ab = ((ab_pred.squeeze() + 1.0) * 127.5).astype(np.uint8)

        # Combine LAB channels
        lab_image = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.uint8)
        lab_image[:, :, 0] = L
        lab_image[:, :, 1:] = ab

        # Convert to RGB
        rgb_image = self.lab_to_rgb(lab_image)
        return rgb_image


def create_sample_dataset(num_samples=100):
    """Create a small sample dataset for testing (generates synthetic colour images)"""
    print("Creating sample dataset...")
    X, y = [], []

    for i in range(num_samples):
        # Create random coloured shapes
        img = np.zeros((256, 256, 3), dtype=np.uint8)

        # Add random coloured rectangles and circles
        for _ in range(np.random.randint(2, 5)):
            # Random colour
            colour = tuple(np.random.randint(0, 255, 3).tolist())

            # Random shape
            if np.random.random() > 0.5:
                # Rectangle
                pt1 = (np.random.randint(0, 200), np.random.randint(0, 200))
                pt2 = (pt1[0] + np.random.randint(20, 56), pt1[1] + np.random.randint(20, 56))
                cv2.rectangle(img, pt1, pt2, colour, -1)
            else:
                # Circle
                center = (np.random.randint(50, 206), np.random.randint(50, 206))
                radius = np.random.randint(10, 30)
                cv2.circle(img, center, radius, colour, -1)

        # Convert to LAB and split
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32) / 255.0
        ab = lab[:, :, 1:].astype(np.float32) / 127.5 - 1.0

        X.append(L[..., np.newaxis])
        y.append(ab)

    return np.array(X), np.array(y)


def train_model(X_train, y_train, X_val, y_val, epochs=50):
    """Train the colourisation model"""
    # Build model
    colourizer = colourisationModel()
    model = colourizer.build_model()
    colourizer.compile_model()

    print("Model built successfully!")
    colourizer.summary()

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )

    return model, history


def visualize_results(model, X_test, y_test, processor, num_samples=4):
    """Visualize colourisation results"""
    predictions = model.predict(X_test[:num_samples])

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Grayscale input
        gray = X_test[i].squeeze()
        axes[i, 0].imshow(gray, cmap='gray')
        axes[i, 0].set_title('Input (Grayscale)')
        axes[i, 0].axis('off')

        # Ground truth
        ground_truth = processor.postprocess_prediction(X_test[i], y_test[i])
        axes[i, 1].imshow(ground_truth)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prediction
        predicted = processor.postprocess_prediction(X_test[i], predictions[i])
        axes[i, 2].imshow(predicted)
        axes[i, 2].set_title('Predicted')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """Main training pipeline"""
    print("Starting Image colourisation System")
    print("=" * 50)

    # Create sample dataset (you can replace this with real image paths)
    print("Step 1: Creating sample dataset...")
    X, y = create_sample_dataset(num_samples=500)  # Create 500 sample images

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Train model
    print("\nStep 2: Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=20)

    # Test and visualize
    print("\nStep 3: Testing and visualisation...")
    processor = DataProcessor()
    visualize_results(model, X_test, y_test, processor)

    # Save model
    model.save('colourisation_model.h5')
    print("\nModel saved as 'colourisation_model.h5'")

    return model, processor


if __name__ == "__main__":
    model, processor = main()