import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load("../ml/unet_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def get_mask(image_rgb: np.ndarray) -> np.ndarray:
    original_size = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (256, 256)) / 255.0
    tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    mask = (output > 0.5).astype(np.uint8)
    return cv2.resize(mask, (original_size[1], original_size[0]))