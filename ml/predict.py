import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Load the trained model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

model.load_state_dict(torch.load("unet_model.pth"))

model = model.to(DEVICE)

model.eval()
# Directory containing processed images
IMAGE_DIR = "data/processed_images"

# Load first 10 images
image_files = os.listdir(IMAGE_DIR)[:10]

for image_name in image_files:

    image_path = os.path.join(IMAGE_DIR, image_name)

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_size = image_rgb.shape[:2]

    # Resize for model
    image_resized = cv2.resize(image_rgb, (256, 256))

    # Normalize
    image_resized = image_resized / 255.0

    # Convert to tensor
    tensor_image = torch.tensor(
        image_resized,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    tensor_image = tensor_image.to(DEVICE)

    # Predict
    with torch.no_grad():

        output = model(tensor_image)

        output = torch.sigmoid(output)

        output = output.squeeze().cpu().numpy()

    # Binary mask
    predicted_mask = (output > 0.5).astype(np.uint8)

    # Resize mask back
    predicted_mask = cv2.resize(
        predicted_mask,
        (original_size[1], original_size[0])
    )

   #display in matplotlib the original image and the segmented image side by side
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap="gray")
    plt.title("Segmentation Image")
    plt.axis("off")

    plt.show()