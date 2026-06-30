import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from autoregressive.models.VTP.vtp.models.vtp_hf import VTPConfig, VTPModel
# from VTP.vtp.models.vtp_hf import VTPConfig, VTPModel



class VTP_Adapter(nn.Module):
    def __init__(self, adapter_size='small', down_sample=1, condition_type=None):
        super(VTP_Adapter, self).__init__()
        print(f"Choose adapter size: {adapter_size}")
        self.model = VTPModel.from_pretrained(f'ControlAR/autoregressive/models/VTP-Small-f16d64').trunk
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
        # x = self.to_patch14(x)
        # x = self.model(x, forward_type="feature")
        x = self.model(x,
            is_training=True,  # Must be True to get full dict output
            use_bottleneck=False,
        )['x_norm_patchtokens']
        # import pdb;pdb.set_trace()
        return x
    

if __name__ == '__main__':
    model = VTP_Adapter('small').cuda()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    inputs = torch.randn(4,3,512,512).cuda()
    outputs = model(inputs)
    print(outputs.shape)