import os
import cv2
import numpy as np
from pycocotools.coco import COCO

# Paths for COCO dataset
ANNOTATION_FILE = "data/annotations/instances_train2017.json"
IMAGE_DIR = "data/train2017"

OUTPUT_IMAGE_DIR = "data/processed_images"
OUTPUT_MASK_DIR = "data/processed_masks"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# Create object of COCO dataset
coco = COCO(ANNOTATION_FILE)

# Get ID for person class
cat_ids = coco.getCatIds(catNms=['person'])

# Get image IDs containing persons
img_ids = coco.getImgIds(catIds=cat_ids)

# Use only first 500 images for training
img_ids = img_ids[:500]

print(f"Total selected images: {len(img_ids)}")

for idx, img_id in enumerate(img_ids):

    # Load image info
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGE_DIR, img_info['file_name'])

    # Read image
    image = cv2.imread(img_path)

    if image is None:
        continue

    # Resize image
    image = cv2.resize(image, (256, 256))

    # Create empty mask
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    # Load annotations
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)

    # Draw masks for model to predict and learn from it
    for ann in anns:
        mask = np.maximum(mask, coco.annToMask(ann) * 255)

    # Resize mask
    mask = cv2.resize(mask, (256, 256))

    # Save image and mask
    image_save_path = os.path.join(OUTPUT_IMAGE_DIR, f"{img_id}.jpg")
    mask_save_path = os.path.join(OUTPUT_MASK_DIR, f"{img_id}.png")

    cv2.imwrite(image_save_path, image)
    cv2.imwrite(mask_save_path, mask)

    print(f"Processed {idx+1}/500")

print("Dataset preparation complete!")