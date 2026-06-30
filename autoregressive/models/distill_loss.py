from transformers import AutoImageProcessor, Dinov2Model, AutoModel
from transformers import AutoProcessor, CLIPSegVisionModel, CLIPVisionModel
import torch
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np

# dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
# image = dataset["test"]["image"][0]
# image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
# image = np.random.uniform(0, 1, (8, 512, 512, 3)).astype(np.float32)
# image = torch.tensor(image)
# inputs = image_processor(image, return_tensors="pt")

class Distill_Loss(torch.nn.Module):

    def __init__(self, precision, device):
        super().__init__()
        self.precision = precision
        self.device = device
        self.model = AutoModel.from_pretrained("ControlAR/autoregressive/models/dinov2-small").to(self.precision).to(self.device)

    def to_patch14(self, input):
        H, W = input.shape[2:]
        new_H = (H // 16) * 14
        new_W = (W // 16) * 14
        output = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='bicubic', align_corners=True)
        return output
    
    
    def compute_distill_loss(self, edited_img, condition_feature, model=None):

        edited_img = self.to_patch14(edited_img)
        with torch.no_grad():
            if model is None:
                edited_img_feature = self.model(edited_img.to(self.device))
                last_hidden_states = edited_img_feature.last_hidden_state[:, 1:].detach()
            else:
                edited_img_feature = model.model(edited_img.to(self.device))
                last_hidden_states = edited_img_feature.last_hidden_state[:, 1:].detach()
        
        distill_loss = F.mse_loss(condition_feature, last_hidden_states)

        return distill_loss