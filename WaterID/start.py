import os
import cv2
import numpy as np
import pandas as pd
import random
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
from torchmetrics import JaccardIndex

warnings.filterwarnings("ignore")





# Data directories
water_dir = 'water'
no_water_dir = 'no_water'

# Load the new metadata CSV file
metadata_df = pd.read_csv('new_metadata.csv')
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

# Perform 90/10 split for train / val
valid_df = metadata_df.sample(frac=0.1, random_state=42)
train_df = metadata_df.drop(valid_df.index)
print(len(train_df))
print(len(valid_df))

class_names = ['no_water', 'water']
class_rgb_values = [(0, 0, 0), (0, 0, 255)]

def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize=20)
        
        if len(image.shape) == 2:  # Grayscale image
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    plt.show()

class LandCoverDataset(torch.utils.data.Dataset):
    def __init__(self, df, augmentation=None, preprocessing=None):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)  # Reading mask as grayscale
        
        # Ensure mask values are binary (0 or 1)
        mask = (mask > 127).astype(np.float32)  # Assuming original mask values are 0 and 255

        # Debug prints
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")

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
    
def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=1024, width=1024, always_apply=True),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    train_transform = [
        album.CenterCrop(height=1024, width=1024, always_apply=True),
    ]
    return album.Compose(train_transform)

# Initialize dataset with augmentation and preprocessing
dataset = LandCoverDataset(
    train_df, 
    augmentation=get_training_augmentation(),
    preprocessing=None,
)
random_idx = random.randint(0, len(dataset)-1)
image, mask = dataset[2]

print("Created dataset")

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)

augmented_dataset = LandCoverDataset(
    train_df, 
    augmentation=get_training_augmentation(),
)

random_idx = random.randint(0, len(augmented_dataset)-1)

print("finished augmentation")

ENCODER = 'resnet50'
ENCODER_WEIGHTS = None
CLASSES = class_names
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

# Create segmentation model with pretrained encoder
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

print("Creating train and val datasets")

# Initialize preprocessing function if encoder weights are not None
if ENCODER_WEIGHTS is not None:
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
else:
    preprocessing_fn = None

# Get train and val dataset instances
train_dataset = LandCoverDataset(
    train_df, 
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

valid_dataset = LandCoverDataset(
    valid_df, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

# Get train and val data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 5

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function
from segmentation_models_pytorch.losses import DiceLoss

loss = DiceLoss(mode='binary')

# Define metrics using torchmetrics
iou_metric = JaccardIndex(task="binary", num_classes=2, threshold=0.5).to(DEVICE)  # Binary segmentation

# Define optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.00008),
])

# Define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)

# Load best saved model checkpoint from previous commit (if present)
if os.path.exists('../input/deepglobe-land-cover-classification-deeplabv3/best_model.pth'):
    model = torch.load('../input/deepglobe-land-cover-classification-deeplabv3/best_model.pth', map_location=DEVICE)
    print('Loaded pre-trained DeepLabV3+ model!')

# Training and validation loop
def train_one_epoch(epoch_index, train_loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def validate_one_epoch(valid_loader, model, loss_fn, metric_fn, device):
    model.eval()
    valid_loss = 0.0
    metric_fn.reset()
    with torch.no_grad():
        for batch in valid_loader:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            valid_loss += loss.item()
            metric_fn.update(outputs, masks.type(torch.int32))
    return valid_loss / len(valid_loader), metric_fn.compute().item()

print("Starting training")

if TRAINING:
    start_time = time.time()  # Start timing

    best_iou_score = 0.0
    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch + 1}/{EPOCHS}')

        # Training step
        train_loss = train_one_epoch(epoch, train_loader, model, optimizer, loss, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation step
        val_loss, val_iou = validate_one_epoch(valid_loader, model, loss, iou_metric, DEVICE)
        print(f"Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

        # Save model if a better val IoU score is obtained
        if best_iou_score < val_iou:
            best_iou_score = val_iou
            torch.save(model, './best_model.pth')
            print('Model saved!')

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time // 60:.0f} minutes, {elapsed_time % 60:.2f} seconds")
    
# Load best saved model checkpoint from the current run
if os.path.exists('./best_model.pth'):
    best_model = torch.load('./best_model.pth', map_location=DEVICE)
    print('Loaded DeepLabV3+ model from this run.')

# Create test dataloader to be used with DeepLabV3+ model (with preprocessing operation: to_tensor(...))
test_dataset = LandCoverDataset(
    valid_df, 
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)

test_dataloader = DataLoader(test_dataset)

# Test dataset for visualization (without preprocessing augmentations & transformations)
test_dataset_vis = LandCoverDataset(
    valid_df,
    augmentation=get_validation_augmentation(),
)

# Get a random test image/mask index
random_idx = random.randint(0, len(test_dataset_vis) - 1)
image, mask = test_dataset_vis[random_idx]

sample_preds_folder = 'sample_predictions/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)
    
# Updated prediction loop for binary segmentation
for idx in range(len(test_dataset)):

    image, gt_mask = test_dataset[idx]
    image_vis = test_dataset_vis[idx][0].astype('uint8')
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

    # Predict test image
    pred_mask = best_model(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()

    # Binarize the predicted mask using a threshold of 0.5, since using sigmoid activation
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Prepare stack for saving the composite image
    vis_stack = np.hstack([image_vis, gt_mask * 255, pred_mask * 255])

    # Save composite image
    cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), vis_stack[:, :, ::-1])

    # Visualize original image, ground truth mask, and predicted mask
    visualize(
        original_image=image_vis,
        ground_truth_mask=gt_mask * 255,  # Scale to 0-255 for visualization
        predicted_mask=pred_mask * 255,   # Scale to 0-255 for visualization
    )

# Final evaluation on test data
test_loss, test_iou = validate_one_epoch(test_dataloader, best_model, loss, iou_metric, DEVICE)
print("Evaluation on Test Data: ")
print(f"Mean IoU Score: {test_iou:.4f}")
print(f"Mean Dice Loss: {test_loss:.4f}")
