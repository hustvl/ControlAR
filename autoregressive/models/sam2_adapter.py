from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
import torch.nn as nn
from typing import List, Optional
from autoregressive.models.sam2.image_encoder import ImageEncoder, FpnNeck
from autoregressive.models.sam2.hieradet import Hiera
from autoregressive.models.sam2.position_encoding import PositionEmbeddingSine


class Sam2_Adapter(nn.Module):
    def __init__(self, adapter_size='small'):
        super(Sam2_Adapter, self).__init__()
        print(f"Choose adapter size: {adapter_size}")
        self.trunk = Hiera(embed_dim=96, num_heads=1, stages=[1,2,11,2], global_att_blocks=[7,10,13], window_pos_embed_bkg_spatial_size=[7,7])
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)
        self.neck = FpnNeck(position_encoding=self.position_encoding, d_model=256, backbone_channel_list=[768, 384, 192, 96], fpn_top_down_levels=[2,3],fpn_interp_model='nearest')
        
        self.model = ImageEncoder(trunk=self.trunk, neck=self.neck, scalp=1)
        checkpoint = torch.load(f'ControlAR/autoregressive/models/sam2/sam2.1_hiera_{adapter_size}.pt',weights_only=False, map_location="cpu")
        model_weight = checkpoint["model"]
        image_encoder_weights = {
            k[len("image_encoder."):]: v 
            for k, v in model_weight.items() 
            if k.startswith("image_encoder.")
        }
        
        self.model.load_state_dict(image_encoder_weights, strict=True)

    def forward(self, x):
        x = self.model(x)
        return x['vision_features'].flatten(2,3).permute(0,2,1)


if __name__ == '__main__':
    model = Sam2_Adapter().cuda()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    inputs = torch.randn(4,3,512,512).cuda()
    outputs = model(inputs)
    print(outputs.shape)