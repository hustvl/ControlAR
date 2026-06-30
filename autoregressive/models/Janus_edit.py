import torch
import torch.nn as nn
from autoregressive.models.dinov2_adapter import Dinov2_Adapter
from autoregressive.models.sam2_adapter import Sam2_Adapter
from autoregressive.models.SemanticFusion import SemanticFusionBlock
from autoregressive.models.qwenvl_connector import Qwen3VLVisionConnectorSimple
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, List

class QwenVLEmbedder(nn.Module):
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
        # nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class JanusEdit(nn.Module):
    def __init__(self, args, model=None, language_model=None, gen_head=None, dim=2048):
        super().__init__()
        self.dim = dim
        self.qwenvl_image_size = args.qwenvl_image_size
        self.use_connector = args.use_connector
        self.use_adapter = args.use_adapter
        self.use_fusion = args.use_fusion
        self.adapter_name = args.adapter_name
        self.encoder_name = args.encoder_name
        # self.language_model = language_model
        # self.gen_head = gen_head
        self.model = model
        self.img_token_num = 576
        if self.encoder_name == "dino":
            self.adapter = Dinov2_Adapter(adapter_size=args.adapter_size, condition_type=args.condition_type)
            # self.adapter = EVA_Adapter()
            if args.adapter_size == "small":
                self.adapter_mlp = MLP(384, dim, dim)
            elif args.adapter_size == 'base':
                self.adapter_mlp = MLP(768, dim, dim)
        elif self.encoder_name == "qwen":
            if self.qwenvl_image_size == 512:
                self.qwenvl_norm = RMSNorm(dim, eps=1e-5)
                if self.use_adapter:
                    print(f"使用adapter: {self.adapter_name}")
                    if self.adapter_name == 'dinov2':
                        self.adapter = Dinov2_Adapter(adapter_size=args.adapter_size)
                        if args.adapter_size == 'small':
                            input_dim = 2048#+384
                            query_dim = 384
                        elif args.adapter_size == 'base':
                            input_dim = 2048# + 768
                            query_dim = 768
                    elif self.adapter_name == 'sam2':
                        self.adapter = Sam2_Adapter(adapter_size=args.adapter_size)
                        input_dim = 2048 + 256
                    if self.use_fusion:
                        print("使用fusion")
                        input_dim = 2048
                        self.fusionblock = SemanticFusionBlock(dim=2048, query_dim=query_dim, seq_len=576)
                        


        self.control_strength = nn.Parameter(torch.zeros(1))
        
        
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
            edited_img_indices: torch.Tensor, 
            source_image: torch.Tensor, 
            qwenvl_embedding: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None, 
            lengths: Optional[torch.Tensor] = None, 
            padding_mask: Optional[torch.Tensor] = None, 
        ):
        
        if self.use_adapter:
            source_image_embedding = self.adapter(source_image)
            if self.use_fusion:
                qwenvl_embedding = self.fusionblock(source_image_embedding, qwenvl_embedding, padding_mask)
                condition_embeddings = qwenvl_embedding
            else:
                if self.qwenvl_image_size == 512:
                    B, L, D = qwenvl_embedding.shape
                    qwenvl_embedding = qwenvl_embedding.view(B, 16, 16, D).permute(0,3,1,2)
                    qwenvl_embedding = F.interpolate(qwenvl_embedding,size=(24,24),mode='bilinear',align_corners=False).permute(0,2,3,1).flatten(1,2)

                condition_embeddings = torch.cat((self.qwenvl_mlp(qwenvl_embedding), source_image_embedding), dim=2)
        if self.use_connector:
            condition_embeddings = self.qwenvl_connector(condition_embeddings)
        self.condition_token = condition_embeddings * self.control_strength
       
        inputs_embeds[:,49:625] = inputs_embeds[:,49:625] + self.condition_token

        outputs = self.model.language_model.model(inputs_embeds=inputs_embeds,
                                                  position_ids=position_ids,
                                                  attention_mask=attention_mask,
                                                  use_cache=False,
                                                  past_key_values= None)

        hidden_states = outputs.last_hidden_state

        # gather from img tokens
 
        img_hidden_states = hidden_states[:, 49:625]
        logits = self.model.gen_head(img_hidden_states)
        b, s = edited_img_indices.shape
        ce_loss = F.cross_entropy(logits.reshape(b * s, -1), edited_img_indices.reshape(-1), reduction='mean')
        return ce_loss, logits
        