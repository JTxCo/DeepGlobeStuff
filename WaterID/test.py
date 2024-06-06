import torch
import segmentation_models_pytorch as smp

# Initialize model
model = smp.DeepLabV3Plus(encoder_name='resnet50', classes=1, activation='sigmoid')

# Dummy data mimicking your actual data (CPU Example)
dummy_images = torch.randn((1, 3, 1024, 1024))
dummy_masks = torch.randn((1, 1, 1024, 1024))

try:
    outputs = model(dummy_images)  # Minimal execution
    print("Model forward pass successful, outputs shape: ", outputs.shape)
except Exception as e:
    print(f"Exception encountered during model execution: {e}")
