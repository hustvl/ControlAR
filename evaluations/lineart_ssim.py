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
from autoregressive.test.metric import RMSE, SSIM
import torch.nn.functional as F
from condition.hed import HEDdetector
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from condition.lineart import LineArt
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

model = LineArt()
model.load_state_dict(torch.load('condition/ckpts/model.pth', map_location=torch.device('cpu')))
model.cuda()
# Define the dataset and data loader
# img_dir = 'sample/lineart/visualization'
# img_dir = '/data/vjuicefs_ai_camera_album_ql/11171709/Lumina-mGPT/lumina_mgpt/sample/multigen/lineart'
img_dir = '/data/vjuicefs_ai_camera_album_ql/11171709/OmniGen/sample/multigen/lineart'
# label_dir = 'sample/lineart/annotations'
label_dir = '/data/vjuicefs_ai_camera_album_ql/public_data/11171709/model_weight/ControlAR/sample/multigen/lineart/annotations'
dataset = ImageDataset(img_dir, label_dir)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).cuda()
ssim_score = []
with torch.no_grad():
    for images, labels in tqdm(data_loader):
        images = images.permute(0,3,1,2).cuda()
        outputs = model(images.float())*255
        predicted_hed = outputs
        labels = labels[:, 0:1, :, :].cuda()
        ssim_score.append(ssim((predicted_hed/255.0).clip(0,1), (1-(labels/255.0)).clip(0,1)))

print(f'ssim: {torch.stack(ssim_score).mean()}')

with open('/data/vjuicefs_ai_camera_album_ql/11171709/ControlAR/evaluations/lineart.txt', 'a') as f:
    f.write(str(torch.stack(ssim_score).mean().item()) + '\n')  # 转为字符串，并添加换行符以便每行一个值