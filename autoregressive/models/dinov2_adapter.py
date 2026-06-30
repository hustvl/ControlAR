from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
import torch.nn as nn


class Dinov2_Adapter(nn.Module):
    def __init__(self, adapter_size='small', down_sample=1, condition_type=None):
        super(Dinov2_Adapter, self).__init__()
        print(f"Choose adapter size: {adapter_size}")
        self.model = AutoModel.from_pretrained(f'autoregressive/models/dinov2-{adapter_size}')
        self.down_sample = down_sample
        self.condition_type = condition_type
    
    def to_patch14(self, input):
        H, W = input.shape[2:]
        if self.down_sample > 1:
            H = H // self.down_sample
            W = W // self.down_sample
        new_H = (H // 16) * 14
        new_W = (W // 16) * 14
        if self.condition_type in ['canny', 'seg']:
            output = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='nearest')#, align_corners=True)
        else:
            output = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='bilinear', align_corners=True) # depth, lineart, hed
        return output
        
    def forward(self, x):
        x = self.to_patch14(x)
        x = self.model(x)
        return x.last_hidden_state[:, 1:]


if __name__ == '__main__':
    model = Dinov2_Adapter('small').cuda()
    inputs = torch.randn(4,3,512,512).cuda()
    outputs = model(inputs)
    print(outputs.shape)