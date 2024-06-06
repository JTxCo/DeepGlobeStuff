import glob
import cupy as np
import cv2
from skimage import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths to the data directories
water_image_path = 'water/*.tif'
no_water_image_path = 'no_water/*.tif'
water_mask_path = 'water/*.png'
no_water_mask_path = 'no_water/*.png'

# Function to load all images and masks from directories
def load_data(image_paths, mask_paths):
    images = []
    masks = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        # Read the image (TIFF)
        image = io.imread(image_path)
        # Read the mask (PNG) and binarize it
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(int)  # Convert to binary labels (0 and 1)
        images.append(image)
        masks.append(mask)
    return images, masks

# Get all file paths
water_image_files = glob.glob(water_image_path)
no_water_image_files = glob.glob(no_water_image_path)
water_mask_files = glob.glob(water_mask_path)
no_water_mask_files = glob.glob(no_water_mask_path)

print("Got all file paths")
# Load all data
water_images, water_masks = load_data(water_image_files, water_mask_files)
no_water_images, no_water_masks = load_data(no_water_image_files, no_water_mask_files)
print("Loaded all data")
# Combine the data
all_images = water_images + no_water_images
all_masks = water_masks + no_water_masks

# Function to extract features from each image
def extract_features(images):
    features = []
    for image in images:
        h, w, c = image.shape
        for y in range(h):
            for x in range(w):
                feature = list(image[y, x])  # RGB values
                feature.append(x / w)  # Normalized x coordinate
                feature.append(y / h)  # Normalized y coordinate
                features.append(feature)
    return np.array(features)

print("Extracting features")
# Extract features and flatten masks
all_features = extract_features(all_images)
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