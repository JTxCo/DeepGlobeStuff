import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step 1: Load the image
image = cv2.imread('no_water/208695_image.tif')
if image is None:
    print("Error: Unable to load image.")
    exit()

# Step 2: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply a binary threshold
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 4: Remove small noise through morphological operations
kernel = np.ones((3, 3), np.uint8)
# Remove noise
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Finding sure foreground area using distance transform and thresholding
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Step 5: Marker labelling
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Label markers
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0 but 1
markers = markers + 1

# Mark the region of unknown with zero
markers[unknown == 0] = 0

# Step 6: Apply the watershed algorithm
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]  # Mark boundaries with red

# Step 7: Visualize the result
plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Watershed Result')
plt.subplot(222), plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB)), plt.title('Binary Image')
plt.subplot(223), plt.imshow(cv2.cvtColor(sure_fg, cv2.COLOR_BGR2RGB)), plt.title('Sure Foreground')
plt.subplot(224), plt.imshow(cv2.cvtColor(sure_bg, cv2.COLOR_BGR2RGB)), plt.title('Sure Background')
plt.show()
