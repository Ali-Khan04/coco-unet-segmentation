import torch
import segmentation_models_pytorch as smp

# Check if graphics card is detected
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

#U Net : 
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

print("\nUNet model loaded")