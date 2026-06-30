
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torchvision.transforms as transforms
from PIL import Image

img1 = Image.open('autoregressive/test/label.png').convert('L')  # 
img2 = Image.open('autoregressive/test/pred.png').convert('L')

to_tensor = transforms.ToTensor()
img1_tensor = to_tensor(img1).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
img2_tensor = to_tensor(img2).unsqueeze(0)

img1_tensor = img1_tensor.float()
img2_tensor = img2_tensor.float()

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

ms_ssim_score = ms_ssim(img1_tensor, img2_tensor)

print("MS-SSIM:", ms_ssim_score.item())