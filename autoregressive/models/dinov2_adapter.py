from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
import torch.nn as nn


class Dinov2_Adapter(nn.Module):
    def __init__(self, input_dim=1, output_dim=768, attention=False, pool=False, nheads=8, dropout=0.1, adapter_size='small', condition_type='canny'):
        super(Dinov2_Adapter, self).__init__()
        print(f"Choose adapter size: {adapter_size}")
        print(f"condition type: {condition_type}")
        self.model = AutoModel.from_pretrained(f'autoregressive/models/dinov2-{adapter_size}')
        self.condition_type = condition_type
    
    def to_patch14(self, input):
        H, W = input.shape[2:]
        new_H = (H // 16) * 14
        new_W = (W // 16) * 14
        if self.condition_type in ['canny', 'seg']:
            output = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='nearest')#, align_corners=True)  canny, seg
        else:
            output = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='bicubic', align_corners=True) # depth, lineart, hed
        return output
        
    def forward(self, x):
        x = self.to_patch14(x)
        x = self.model(x)
        return x.last_hidden_state[:, 1:]


if __name__ == '__main__':
    model = Dinov2_Adapter().cuda()
    inputs = torch.randn(4,3,512,512).cuda()
    outputs = model(inputs)
    print(outputs.shape)