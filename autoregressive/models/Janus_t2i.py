import torch
import torch.nn as nn
from autoregressive.models.dinov2_adapter import Dinov2_Adapter
from autoregressive.models.sam2_adapter import Sam2_Adapter
from autoregressive.models.SemanticFusion import SemanticFusionBlock
from autoregressive.models.qwenvl_connector import Qwen3VLVisionConnectorSimple
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, List

class ConditionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=1024):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", torch.zeros(token_num, in_channels) / in_channels ** 0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption, drop_ids

    def forward(self, caption, train=False, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption, drop_ids = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        if (train and use_dropout) or (force_drop_ids is not None):
            return embeddings,drop_ids
        else:
            return embeddings


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        
        # nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Janus_models(nn.Module):
    def __init__(self, args, model=None, language_model=None, gen_head=None, dim=2048):
        super().__init__()
        self.dim = dim
        self.adapter_size = args.adapter_size
        self.condition_type = args.condition_type
        self.model = model
        self.text_length = args.text_length
        
        self.adapter = Dinov2_Adapter(adapter_size=self.adapter_size, condition_type=self.condition_type)
        if self.adapter_size == 'small':
            input_dim = 384
        elif self.adapter_size == 'base':
            input_dim = 768
        self.condition_mlp = ConditionEmbedder(input_dim, self.dim, 0.0)
        
        self.img_token_num = 576

        self.control_strength = nn.Parameter(torch.zeros(1))
        self.register_buffer("uncond_condition_embedding", nn.Parameter(torch.zeros(576, 2048)))
        
        
    def _init_weights(self, module):
        std = 0.02#self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(self,   
            inputs_embeds: torch.Tensor, 
            target_indices: torch.Tensor, 
            source_image: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None, 
            lengths: Optional[torch.Tensor] = None, 
            drop_ids: Optional[torch.Tensor] = None, 
        ):
        
        source_image_embedding = self.adapter(source_image)
        condition_embeddings = self.condition_mlp(source_image_embedding)
        
        force_drop_ids = drop_ids == 1
        condition_embeddings = torch.where(force_drop_ids[:, None, None], self.uncond_condition_embedding, condition_embeddings)

        self.condition_token = condition_embeddings# * self.control_strength
        
        # for i, length in enumerate(lengths):
        #     inputs_embeds[i, int(length.item())-1:int(length.item())-1+576] = inputs_embeds[i, int(length.item())-1:int(length.item())-1+576] + self.condition_token[i]
        
        inputs_embeds[:,self.text_length-1:self.text_length-1+576] = inputs_embeds[:,self.text_length-1:self.text_length-1+576] + self.condition_token

        outputs = self.model.language_model.model(inputs_embeds=inputs_embeds,
                                                  position_ids=position_ids,
                                                  attention_mask=attention_mask,
                                                  use_cache=False,
                                                  past_key_values= None)

        hidden_states = outputs.last_hidden_state

        img_hidden_states = hidden_states[:, self.text_length-1:self.text_length-1+576]
        


        
        logits = self.model.gen_head(img_hidden_states)
        b, s = target_indices.shape
        ce_loss = F.cross_entropy(logits.reshape(b * s, -1), target_indices.reshape(-1), reduction='mean')
        return ce_loss, logits
        