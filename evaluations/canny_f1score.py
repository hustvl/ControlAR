import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
current_directory = os.getcwd()
sys.path.append(current_directory)
from autoregressive.test.metric import RMSE, SSIM, F1score
import torch.nn.functional as F
from condition.hed import HEDdetector
from condition.canny import CannyDetector
from torchmetrics.classification import BinaryF1Score
# Define a dataset class for loading image and label pairs
class ImageDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path))
        return torch.from_numpy(image), torch.from_numpy(label).permute(2, 0, 1)

model = CannyDetector()
# Define the dataset and data loader
img_dir = 'sample/multigen/canny/visualization'
label_dir = 'sample/multigen/canny/annotations'
dataset = ImageDataset(img_dir, label_dir)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# Instantiate the metric
f1score = BinaryF1Score()
f1 = []
i = 0
with torch.no_grad():
    for images, labels in tqdm(data_loader):
        i += 1
        images = images
        outputs = []
        for img in images:   
            outputs.append(model(img))
        # Move predictions and labels to numpy for RMSE calculation
        predicted_canny = outputs
        labels = labels[:, 0, :, :].numpy() # Assuming labels are in Bx1xHxW format
          
        for pred, label in zip(predicted_canny, labels):
            pred[pred == 255] = 1
            label[label == 255] = 1
            f1.append(f1score(torch.from_numpy(pred).flatten(), torch.from_numpy(label).flatten()).item())

print(f'f1score: {np.array(f1).mean()}')