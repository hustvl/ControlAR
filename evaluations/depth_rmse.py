import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys
current_directory = os.getcwd()
sys.path.append(current_directory)
from autoregressive.test.metric import RMSE
import torch.nn.functional as F
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
        image = Image.open(img_path).convert("RGB")
        label = np.array(Image.open(label_path))

        return np.array(image), torch.from_numpy(label).permute(2, 0, 1)

# Instantiate the model and processor
processor = DPTImageProcessor.from_pretrained("condition/ckpts/dpt_large")
model = DPTForDepthEstimation.from_pretrained("condition/ckpts/dpt_large").cuda()

# Define the dataset and data loader
img_dir = 'sample/multigen/depth/visualization'
label_dir = 'sample/multigen/depth/annotations'
dataset = ImageDataset(img_dir, label_dir)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# Instantiate the metric
metric = RMSE()

# Perform inference on batches and calculate RMSE
model.eval()
rmse = []
with torch.no_grad():
    for images, labels in tqdm(data_loader):
        inputs = processor(images=images, return_tensors="pt", size=(512,512)).to('cuda:0')
        outputs = model(**inputs)
        
        predicted_depth = outputs.predicted_depth
        predicted_depth = predicted_depth.squeeze().cpu()
        labels = labels[:, 0, :, :]
        
        for pred, label in zip(predicted_depth, labels):
            # Preprocess predicted depth for fair comparison
            pred = (pred * 255 / pred.max())
            per_pixel_mse = torch.sqrt(F.mse_loss(pred.float(), label.float()))
            rmse.append(per_pixel_mse)
print(np.array(rmse).mean())