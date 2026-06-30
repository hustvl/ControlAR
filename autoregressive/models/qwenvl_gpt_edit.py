# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
# from autoregressive.models.vit_adapter import ViT_Adapter
from autoregressive.models.dinov2_adapter import Dinov2_Adapter


class QwenVLGPTEdit(nn.Module):
    def __init__(self, args, qwenvl_model, processor, gpt_model):
        super(QwenVLGPTEdit, self).__init__()
        self.use_code = args.use_code
        self.qwenvl_model = qwenvl_model
        self.processor = processor
        self.gpt_model = gpt_model
        self.qwenvl_image_size = args.qwenvl_image_size
        self.qwenvl_tokens_length = (self.qwenvl_image_size // 32) ** 2
        
    def forward(self, captions, images, input_txt_embs, edited_img_indices, attn_mask, condition_img=None, source_image=None, target_image=None, qwenvl_embedding=None, padding_mask=None):
        if self.use_code:
            _, loss, condition_feature = self.gpt_model(cond_idx=input_txt_embs, idx=edited_img_indices[:,:-1], targets=edited_img_indices, mask=attn_mask[:, :, :-1,:-1], qwenvl_embedding=qwenvl_embedding, source_image=source_image, padding_mask=padding_mask)
        else:
            device = next(self.qwenvl_model.parameters()).device
            prompts = []
            for caption, image in zip(captions, images):
                # caption = f"Given an image editing instruction: '{caption},' please locate the object that needs to be edited within the image."
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": caption},
                        {
                        "type": "image",
                        },  
                    ],
                }]
                prompts.append(self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                ))
            qwenvl_inputs = self.processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            qwenvl_inputs = qwenvl_inputs.to(device)
            qwenvl_outputs = self.qwenvl_model.forward(
                **qwenvl_inputs,
                output_hidden_states=True
            )
            qwenvl_outputs_embedding = qwenvl_outputs['hidden_states'][-1][:, -259:]#[:,-self.qwenvl_tokens_length:]
            _, loss, condition_feature = self.gpt_model(cond_idx=input_txt_embs, idx=edited_img_indices[:,:-1], targets=edited_img_indices, mask=attn_mask[:, :, :-1,:-1], qwenvl_embedding=qwenvl_outputs_embedding, source_image=source_image)

        return loss, condition_feature
