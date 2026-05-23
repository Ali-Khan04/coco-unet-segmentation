import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader
from torch import nn, optim


IMAGE_DIR = "data/processed_images"
MASK_DIR = "data/processed_masks"

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



#dataset class to load the images and masks for training


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name.replace(".jpg", ".png"))

        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize image
        image = image / 255.0

        # Convert mask to binary
        mask = mask / 255.0

        # Convert to tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask
#dataset and dataloader to load the images and masks for training
dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

#load unet
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load("unet_model.pth", map_location=DEVICE))
#move model to gpu 
model = model.to(DEVICE)

loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
#training loop
print("Training started...\n")

for epoch in range(EPOCHS):

    epoch_loss = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        # Backpropagation
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_fn(outputs, masks)



        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "unet_model.pth")

print("\nModel saved as unet_model.pth")