import cv2

# Load the grayscale image
import numpy as np
import base64
import requests

def load_image(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Unable to load image from {image_path}")
    return image

def apply_otsu_threshold(image):
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def save_image(image, output_path):
    # Save the binary image to a file
    cv2.imwrite(output_path, image)
    print(f"Binary image saved as {output_path}")

def upload_to_imgur(image_path, client_id):
    # Read the image file
    with open(image_path, 'rb') as file:
        img_data = file.read()

    # Encode the image to base64
    img_base64 = base64.b64encode(img_data).decode('utf-8')


    if response.status_code == 200:
        print("Image uploaded successfully.")
        print("View it at:", response.json()['data']['link'])
    else:
        print("Failed to upload image.", response.status_code, response.text)

# Path to your image
input_image_path = '/home/stu686191/code/DeepGlobeStuff/WaterID/no_water/208695_image.tif' 
binary_image_path = 'binary_image.tif'  # Save location for the binary image

# Load the image
image = load_image(input_image_path)

# Apply Otsu's thresholding
binary_image = apply_otsu_threshold(image)

# Save the binary image
save_image(binary_image, binary_image_path)


# Check if the image loaded successfully
if image is None:
    print("Error: Unable to load image.")
    exit()

# Apply Otsu's thresholding
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save the binary image to a file
cv2.imwrite('binary_image.png', binary_image)

print("Binary image saved as binary_image.png")

