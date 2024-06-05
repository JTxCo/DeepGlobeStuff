import os, cv2
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pandas as pd
import albumentations as album
import numpy as np

# Set device (using CPU)
DEVICE = torch.device("cpu")

# Define model parameters
ENCODER = 'resnet18'
ENCODER_WEIGHTS = None  # Initialize without pretrained weights
CLASSES = ['no_water', 'water']
ACTIVATION = 'sigmoid'

print("Creating model...")
try:
    print("in try block")
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    ).to(DEVICE)
    print("Model created successfully!")
except Exception as e:
    print(f"Error during model creation: {e}")
    raise e


# Load the new metadata CSV file
metadata_df = pd.read_csv('new_metadata.csv')
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

# Split dataset
valid_df = metadata_df.sample(frac=0.1, random_state=42)
train_df = metadata_df.drop(valid_df.index)
print(len(train_df))
print(len(valid_df))

# Dataset and DataLoader classes
# Assuming get_training_augmentation and other necessary functions are defined

class LandCoverDataset(torch.utils.data.Dataset):
    def __init__(self, df, augmentation=None, preprocessing=None):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.augmentation:
            try:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            except Exception as e:
                print(f"Augmentation error for index {i}: {e}")
                raise e

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask

    def __len__(self):
        return len(self.image_paths)
    
dataset = LandCoverDataset(
    train_df,
    augmentation=album.Compose([album.RandomCrop(height=1024, width=1024, always_apply=True)]),
)

print("Created dataset")

# Initialize preprocessing function if encoder weights are not None
if ENCODER_WEIGHTS is not None:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
else:
    preprocessing_fn = None

train_loader = DataLoader(
    LandCoverDataset(train_df, augmentation=album.Compose([album.RandomCrop(height=1024, width=1024, always_apply=True)])),
    batch_size=4, shuffle=True, num_workers=0
)

print("finished augmentation")
