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
import cv2
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
        # label_path = os.path.join(self.label_dir, f'{int(os.path.splitext(self.images[idx])[0]):06d}.png')

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path))#.resize((512,512)))
        return torch.from_numpy(image), torch.from_numpy(label).permute(2, 0, 1)

model = CannyDetector()
# Define the dataset and data loader
# img_dir = 'sample/canny/visualization'
# img_dir = '/data/vjuicefs_ai_camera_album_ql/11171709/Lumina-mGPT/lumina_mgpt/sample/multigen/canny'
img_dir = '/data/vjuicefs_ai_camera_album_ql/11171709/OmniGen/sample/multigen/canny'
# img_dir = '/data/vjuicefs_ai_camera_album_ql/11171709/EditAR/models/editar_release/editar_release/canny/visualization/txt_3.0'
# label_dir = 'sample/canny/annotations'
# label_dir = '/data/vjuicefs_sz_cv_v2/11171709/sz_cv_v2/ControlAR_github/sample/multigen/canny/annotations'
label_dir = '/data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/sample/multigen/canny/annotations'
# label_dir = '/data/vjuicefs_ai_camera_album_ql/public_data/11171709/data/MultiGen20M/val/image'
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
        # import pdb;pdb.set_trace()
        labels = labels[:, 0, :, :].numpy() # Assuming labels are in Bx1xHxW format
        
        # Labels = [] 
        # for img in labels:   
        #     Labels.append(model(img))
        # labels = Labels
        
        for pred, label in zip(predicted_canny, labels):
            # pred[pred < 255] = 0
            pred[pred == 255] = 1
            # label[label<255] = 0
            label[label == 255] = 1
            f1.append(f1score(torch.from_numpy(pred).flatten(), torch.from_numpy(label).flatten()).item())

print(f'f1score: {np.array(f1).mean()}')

with open('/data/vjuicefs_ai_camera_album_ql/11171709/ControlAR/evaluations/canny.txt', 'a') as f:
    f.write(str(np.array(f1).mean()) + '\n')  # 转为字符串，并添加换行符以便每行一个值