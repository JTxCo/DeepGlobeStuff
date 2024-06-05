import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

import segmentation_models_pytorch as smp

# data is in the water and no_water directories in the current directory
water_dir = 'water'
no_water_dir = 'no_water'

# Load the new metadata CSV file
metadata_df = pd.read_csv('new_metadata.csv')
# metadata_df = metadata_df[metadata_df['split']=='train']
# metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
# metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
# metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
# Shuffle DataFrame
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

# Perform 90/10 split for train / val
valid_df = metadata_df.sample(frac=0.1, random_state=42)
train_df = metadata_df.drop(valid_df.index)
print(len(train_df))
print(len(valid_df))
