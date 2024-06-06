import os
import glob

# Get the current working directory
#cwd = os.getcwd()
cwd = '/water/'
# Define the pattern for image files
image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tiff"]

# Find all image files in the CWD matching the patterns;
image_files = []
for pattern in image_patterns:
    image_files.extend(glob.glob(os.path.join(cwd, pattern)))

# Print out the list of image files
for image_file in image_files:
    print(image_file)
