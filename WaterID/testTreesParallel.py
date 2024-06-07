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

# Function to load all images and masks from directories
def load_image_and_mask(paths):
    image_path, mask_path = paths
    image = io.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(int)  # Convert to binary labels (0 and 1)
    return image, mask

# Get all file paths
water_image_files = glob.glob(water_image_path)
no_water_image_files = glob.glob(no_water_image_path)
water_mask_files = glob.glob(water_mask_path)
no_water_mask_files = glob.glob(no_water_mask_path)

print("Got all file paths")

# Load all data using ThreadPoolExecutor
all_image_mask_paths = list(zip(water_image_files, water_mask_files)) + list(zip(no_water_image_files, no_water_mask_files))

def load_data_in_parallel(paths):
    images, masks = [], []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image_and_mask, path) for path in paths]
        for future in as_completed(futures):
            img, msk = future.result()
            images.append(img)
            masks.append(msk)
    return images, masks

print("Loading data in parallel")
water_images, water_masks = load_data_in_parallel(zip(water_image_files, water_mask_files))
no_water_images, no_water_masks = load_data_in_parallel(zip(no_water_image_files, no_water_mask_files))
print("Loaded all data")

# Combine the data
all_images = water_images + no_water_images
all_masks = water_masks + no_water_masks

# Function to extract features from each image
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

def extract_features_in_parallel(images):
    features = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_single_image_features, img) for img in images]
        for future in as_completed(futures):
            features.extend(future.result())
    return np.array(features)

print("Extracting features in parallel")
# Extract features and flatten masks
all_features = extract_features_in_parallel(all_images)
all_labels = np.array([label for mask in all_masks for label in mask.flatten()])
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
