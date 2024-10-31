from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
import torch.nn as nn


class ViT_Adapter(nn.Module):
    def __init__(self, input_dim=3, output_dim=768, attention=False, pool=False, nheads=8, dropout=0.1):
        super(ViT_Adapter, self).__init__()
        self.model = AutoModel.from_pretrained('autoregressive/models/vit-small')
        
    def forward(self, x):
        x = self.model(x,interpolate_pos_encoding=True)
        return x.last_hidden_state[:, 1:]


if __name__ == '__main__':
    model = ViT_Adapter().cuda()
    import pdb;pdb.set_trace()
    print(sum(p.numel() for p in model.parameters()))
    inputs = torch.randn(4,3,512,512).cuda()

    outputs = model(inputs)

    print(outputs.shape)