# This is an improved version and model of HED edge detection with Apache License, Version 2.0.
# Please use this implementation in your products
# This implementation may produce slightly different results from Saining Xie's official implementations,
# but it generates smoother edges and is more suitable for ControlNet as well as other image-to-image translations.
# Different from official models and other implementations, this is an RGB-input model (rather than BGR)
# and in this way it works better for gradio's RGB protocol

import os
import cv2
import torch
import numpy as np
from torch.nn.parallel import DataParallel
from einops import rearrange
from condition.utils import annotator_ckpts_path
import torch.nn.functional as F

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class HEDdetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
        modelpath = os.path.join(annotator_ckpts_path, "ControlNetHED.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)
        self.netNetwork = ControlNetHED_Apache2().float()#.to(self.device).eval()
        self.netNetwork.load_state_dict(torch.load(modelpath))

    def __call__(self, input_image):
        """
        input: tensor (B,C,H,W)
        output: tensor (B,H,W)
        """
        B, C, H, W = input_image.shape
        image_hed = input_image

        edges = self.netNetwork(image_hed)
        edges = [F.interpolate(e, size=(H, W), mode='bilinear', align_corners=False).squeeze(1) for e in edges]
        edges = torch.stack(edges, dim=1)
        edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=1)))
        edge = (edge * 255.0).clamp(0, 255)

        return edge


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import torch.nn.functional as F
    device = torch.device('cuda')
    apply_hed = HEDdetector().to(device).eval()
    img = cv2.imread('condition/dragon_1024_512.jpg')
    H,W = img.shape[:2]
    resize_img = cv2.resize(img,(512,1024))
    detected_map = apply_hed(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda())
    resize_detected_map = apply_hed(torch.from_numpy(resize_img).permute(2,0,1).unsqueeze(0).cuda())
    cv2.imwrite('condition/example_hed_resize.jpg', resize_detected_map[0].cpu().detach().numpy())
    resize_detected_map = F.interpolate(resize_detected_map.unsqueeze(0).to(torch.float32), size=(H,W), mode='bilinear', align_corners=False, antialias=True)
    print(abs(detected_map - resize_detected_map).sum())
    print(img.shape, img.max(),img.min(),detected_map.shape, detected_map.max(),detected_map.min())
    cv2.imwrite('condition/example_hed.jpg', detected_map[0].cpu().detach().numpy())
    cv2.imwrite('condition/example_hed_resized.jpg', resize_detected_map[0,0].cpu().detach().numpy())