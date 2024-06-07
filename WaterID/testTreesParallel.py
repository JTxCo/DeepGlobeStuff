import glob
import numpy as np
import cv2
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths to the data directories
water_image_path = 'water/*.tif'
no_water_image_path = 'no_water/*.tif'
water_mask_path = 'water/*.png'
no_water_mask_path = 'no_water/*.png'

# Function to load image and mask
def load_image_and_mask(paths):
    image_path, mask_path = paths
    # Read the image (TIFF)
    image = io.imread(image_path)
    # Read the mask (PNG) and binarize it
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(int)  # Convert to binary labels (0 and 1)
    return image, mask

# Load data in batches
def load_data_in_batches(paths, batch_size=10):
    images = []
    masks = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        print(f"Loading batch {i // batch_size + 1}/{(len(paths) + batch_size - 1) // batch_size}")

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_image_and_mask, batch_paths))
        for image, mask in results:
            images.append(image)
            masks.append(mask)
    return images, masks

# Get all file paths
water_image_files = glob.glob(water_image_path)[:10]
no_water_image_files = glob.glob(no_water_image_path)[:5]
water_mask_files = glob.glob(water_mask_path)[:10]
no_water_mask_files = glob.glob(no_water_mask_path)[:5]

print("Got all file paths")

# Combine water and no_water file paths
all_image_mask_paths = list(zip(water_image_files, water_mask_files)) + list(zip(no_water_image_files, no_water_mask_files))

print("Loading data in batches")
# Load data in batches
images, masks = load_data_in_batches(all_image_mask_paths, batch_size=5)
print("Loaded all data")

# Define function to extract features from a single image
def extract_single_image_features(image):
    features = []
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            feature = list(image[y, x])  # RGB values
            feature.append(x / w)  # Normalized x coordinate
            feature.append(y / h)  # Normalized y coordinate
            features.append(feature)
    return features

# Extract features in batches
def extract_features_in_batches(images, batch_size=5):
    all_features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        print(f"Extracting features from batch {i // batch_size + 1}/{(len(images) + batch_size - 1) // batch_size}")

        with ThreadPoolExecutor() as executor:
            result_batches = list(executor.map(extract_single_image_features, batch_images))
        for batch in result_batches:
            all_features.extend(batch)
    return np.array(all_features)

print("Extracting features in batches")
# Extract features and flatten masks
all_features = extract_features_in_batches(images)
all_labels = np.array([label for mask in masks for label in mask.flatten()])
print("Extracted features")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)

print("Split data into training and testing sets")
print("Training the model")
# Create and train the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Trained the model")

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Optionally, compute confusion matrix or other metrics
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
