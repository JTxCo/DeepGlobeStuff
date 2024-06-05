import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import cv2
import tifffile

# Set of incorrect segmentation maps
incorrect_segm_maps = set([
    '271941_sat.jpg', '730821_sat.jpg', '256189_sat.jpg', '925382_sat.jpg', '458776_sat.jpg',
    '705728_sat.jpg', '33573_sat.jpg', '943463_sat.jpg', '898741_sat.jpg', '715633_sat.jpg',
    '127976_sat.jpg', '541353_sat.jpg', '615420_sat.jpg', '614561_sat.jpg', '516317_sat.jpg',
    '626208_sat.jpg', '751939_sat.jpg', '952430_sat.jpg', '622733_sat.jpg', '870705_sat.jpg',
    '897901_sat.jpg', '682046_sat.jpg', '904606_sat.jpg', '286339_sat.jpg', '749375_sat.jpg',
    '483506_sat.jpg', '834433_sat.jpg', '253691_sat.jpg', '418261_sat.jpg', '387018_sat.jpg',
    '565914_sat.jpg', '7791_sat.jpg', '139482_sat.jpg', '141685_sat.jpg', '362191_sat.jpg',
    '496948_sat.jpg', '507241_sat.jpg', '470446_sat.jpg', '641771_sat.jpg', '10452_sat.jpg',
    '351271_sat.jpg', '318338_sat.jpg', '548686_sat.jpg', '763075_sat.jpg', '127660_sat.jpg',
    '232373_sat.jpg', '350033_sat.jpg', '834900_sat.jpg', '471930_sat.jpg', '987427_sat.jpg',
    '34330_sat.jpg', '585043_sat.jpg', '254565_sat.jpg', '533948_sat.jpg', '605707_sat.jpg',
    '412210_sat.jpg', '419820_sat.jpg', '21717_sat.jpg', '185562_sat.jpg', '556572_sat.jpg',
    '244423_sat.jpg', '617844_sat.jpg', '651537_sat.jpg', '899693_sat.jpg', '864488_sat.jpg',
    '850510_sat.jpg', '309818_sat.jpg', '991758_sat.jpg', '331533_sat.jpg', '125510_sat.jpg',
    '123172_sat.jpg', '949559_sat.jpg', '858771_sat.jpg', '629198_sat.jpg', '2334_sat.jpg',
    '503968_sat.jpg'
])

DATA_DIR = '../deepglobe-land-cover-classification-dataset'

# Read metadata CSV and filter for training data
metadata_df = pd.read_csv(Path(DATA_DIR) / 'metadata.csv')
metadata_df = metadata_df[metadata_df['split'] == 'train']
metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: str(Path(DATA_DIR) / img_pth))
metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: str(Path(DATA_DIR) / img_pth))
metadata_df.head()

# Create directories if not present
pwd = Path().cwd()
(pwd / 'water').mkdir(exist_ok=True)
(pwd / 'no_water').mkdir(exist_ok=True)

# Initialize list to store new CSV rows
new_csv_data = []

def label_water(mask: np.array) -> np.array:
    # Identify water pixels
    water_mask = (
        (mask[:, :, 0] == 0) & 
        (mask[:, :, 1] == 0) & 
        (mask[:, :, 2] == 255)
    )
    return water_mask.astype(np.uint8)

def process_sample(image_path, mask_path, image_id):
    image_filename = Path(image_path).name

    if image_filename in incorrect_segm_maps:
        return
    
    new_image_filename = image_filename.replace('sat', 'image').replace('jpg', 'tif')
    mask_filename = Path(mask_path).name
    
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

    binary_mask = label_water(mask)

    out_dir = 'water' if np.any(binary_mask) else 'no_water'

    image_out_path = str(pwd / out_dir / new_image_filename)
    mask_out_path = str(pwd / out_dir / mask_filename)
    
    tifffile.imwrite(image_out_path, image, photometric='rgb')
    cv2.imwrite(mask_out_path, binary_mask * 255)  # Save the binary mask scaled to 255 for visibility

    # Create relative paths for the new CSV file
    rel_image_out_path = f"{out_dir}/{new_image_filename}"
    rel_mask_out_path = f"{out_dir}/{mask_filename}"
    
    # Add new data to the list for the new CSV file
    new_csv_data.append({
        'image_id': image_id,
        'sat_image_path': rel_image_out_path,
        'mask_path': rel_mask_out_path,
    })

# Process images concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(lambda row: process_sample(row[0], row[1], row[2]), metadata_df[['sat_image_path', 'mask_path', 'image_id']].values),
              total=len(metadata_df)))

# Create a DataFrame from the new CSV data and save it to a new CSV file
new_metadata_df = pd.DataFrame(new_csv_data)
new_metadata_df.to_csv('new_metadata.csv', index=False)

print("Processing complete and new CSV file created.")
