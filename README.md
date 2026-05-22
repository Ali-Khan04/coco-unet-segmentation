# ML-Based Image Segmentation using U-Net

Semantic image segmentation project using a U-Net architecture trained from scratch on a subset of the COCO dataset.

## Features

- Trained U-Net model from scratch
- Person-class semantic segmentation
- COCO dataset preprocessing and mask generation
- GPU-accelerated training using PyTorch and CUDA
- Prediction visualization on unseen images

## Technologies

- Python
- PyTorch
- OpenCV
- COCO Dataset

## Training Results

### Example 1

![Segmentation Result](assets/result1.png)

### Example 2

![Segmentation Result](assets/result2.png)

## Project Structure

```bash
project/
│
├── data/
├── assets/
├── train.py
├── predict.py
├── prepare_dataset.py
├── README.md
```

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```
## Dataset Setup

This project uses a subset of the COCO 2017 dataset for person-class semantic segmentation.

Download the following files from the official COCO dataset website:

- train2017.zip
- annotations_trainval2017.zip

After extraction, place them inside the following structure:

```bash
data/
│
├── train2017/
├── annotations/
│   ├── instances_train2017.json
```

Then run dataset preprocessing:

```bash
python prepare_dataset.py
```

This generates:
- processed_images/
- processed_masks/

used for training the U-Net model.

### Train model

```bash
python train.py
```

### Run prediction

```bash
python predict.py
```
