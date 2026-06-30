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
from autoregressive.models.sam2_adapter import Sam2_Adapter
from autoregressive.models.vtp_adapter import VTP_Adapter
from autoregressive.models.SemanticFusion import SemanticFusionBlock
from autoregressive.models.qwenvl_connector import Qwen3VLVisionConnectorSimple

def get_causal_mask(seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).type(torch.bool)
    mask = mask.masked_fill(mask, float('-inf'))  
    mask = mask.masked_fill(~mask, float(0.0))  
    return mask

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)







@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048
    adapter_size: str = 'small'
    condition_type: str = 'canny'
    encoder_name: str = 'vqgan',
    qwenvl_image_size: int = 1024,
    use_connector: bool = False,
    use_adapter: bool = False,
    use_fusion: bool = False,
    adapter_name: str = 'dinov2',
    use_prefilling: bool = False,
    



#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels, drop_ids

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels,drop_ids = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        if (train and use_dropout) or (force_drop_ids is not None):
            return embeddings,drop_ids
        else:
            return embeddings


class ConditionEmbedder(nn.Module):
    """
    Embeds Condition into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120, vocab_size=16384):
        super().__init__()
        self.cap_proj = MLP(in_features=hidden_size, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", torch.zeros(token_num, hidden_size) / hidden_size ** 0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None, drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            if drop_ids is None:
                drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1

        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding[:caption.shape[1]], caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None, drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids, drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings

#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(120, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob
        self.token_num = token_num

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding[-self.token_num:], caption)
        return caption, drop_ids

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption, drop_ids = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        if (train and use_dropout) or (force_drop_ids is not None):
            return embeddings,drop_ids
        else:
            return embeddings

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


#################################################################################
#                                  GPT Model                                    #
#################################################################################
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


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out



class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.layer_internal = config.n_layer // 1
        self.qwenvl_image_size = config.qwenvl_image_size
        self.use_connector = config.use_connector
        self.use_adapter = config.use_adapter
        self.use_fusion = config.use_fusion
        self.adapter_name = config.adapter_name
        self.use_prefilling = config.use_prefilling
        # self.adapter = Adapter(output_dim=768)
        # self.adapter = ViT_Adapter()
        # self.adapter = DeiT_Adapter()
        self.encoder_name = config.encoder_name
        if self.encoder_name == "dino":
            self.adapter = Dinov2_Adapter(adapter_size=config.adapter_size, condition_type=config.condition_type)
            # self.adapter = EVA_Adapter()
            if config.adapter_size == "small":
                self.adapter_mlp = MLP(384, config.dim, config.dim)
            elif config.adapter_size == 'base':
                self.adapter_mlp = MLP(768, config.dim, config.dim)
        elif self.encoder_name == "qwen":
            if self.qwenvl_image_size == 512:
                self.qwenvl_norm = RMSNorm(config.dim, eps=config.norm_eps)
                if self.use_adapter:
                    print(f"使用adapter: {self.adapter_name}")
                    if self.adapter_name == 'dinov2':
                        self.adapter = Dinov2_Adapter(adapter_size=config.adapter_size)
                        if config.adapter_size == 'small':
                            input_dim = 384
                        elif config.adapter_size == 'base':
                            input_dim = 768
                    elif self.adapter_name == 'sam2':
                        self.adapter = Sam2_Adapter(adapter_size=config.adapter_size)
                        input_dim = 256
                    elif self.adapter_name == 'vtp':
                        self.adapter = VTP_Adapter(adapter_size=config.adapter_size)
                        input_dim = 384
                    if self.use_fusion:
                        print("使用fusion")
                        # input_dim = 1280
                        self.fusionblock = SemanticFusionBlock(query_dim=input_dim, dim=1280)
                    if self.use_prefilling:
                        self.prefilling_adapter = Dinov2_Adapter(adapter_size=config.adapter_size, down_sample=4)
                        self.prefilling_mlp = MLP(in_features=384, hidden_features=config.dim, out_features=config.dim)
                        
                    self.qwenvl_mlp = QwenVLEmbedder(input_dim, config.dim-384, config.class_dropout_prob)
                else:
                    self.qwenvl_mlp = QwenVLEmbedder(2048, config.dim, config.class_dropout_prob)
                if self.use_connector:
                    print("使用connector")
                    self.qwenvl_connector = Qwen3VLVisionConnectorSimple(depth=2, seq_len=1024)
                
            elif self.qwenvl_image_size == 1024:
                self.qwenvl_norm = RMSNorm(config.dim, eps=config.norm_eps)
                if self.use_adapter:
                    print(f"使用adapter: {self.adapter_name}")
                    if self.adapter_name == 'dinov2':
                        self.adapter = Dinov2_Adapter(adapter_size=config.adapter_size)
                        if config.adapter_size == 'small':
                            input_dim = 2048+384 
                        elif config.adapter_size == 'base':
                            input_dim = 2048 + 768
                    elif self.adapter_name == 'sam2':
                        self.adapter = Sam2_Adapter(adapter_size=config.adapter_size)
                        input_dim = 2048 + 256
                    if self.use_fusion:
                        print("使用fusion")
                        input_dim = 1280
                        self.fusionblock = SemanticFusionBlock(dim=1280)
                    if self.use_prefilling:
                        self.prefilling_adapter = Dinov2_Adapter(adapter_size=config.adapter_size, down_sample=4)
                        self.prefilling_mlp = MLP(in_features=input_dim, hidden_features=config.dim, out_features=config.dim)
                        
                    self.qwenvl_mlp = QwenVLEmbedder(input_dim, config.dim, config.class_dropout_prob)
                else:
                    self.qwenvl_mlp = QwenVLEmbedder(2048, config.dim, config.class_dropout_prob)
                if self.use_connector:
                    print("使用connector")
                    self.qwenvl_connector = Qwen3VLVisionConnectorSimple(depth=2, seq_len=1024)
                
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i' or self.model_type == 'edit':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob, token_num=self.cls_token_num)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        
        # self.condition_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        # self.initialize_weights()
        self.condition_token = None
        self.mask = get_causal_mask(256)
        self.global_token = None

        self.control_strength = nn.Parameter(torch.zeros(1))
        
        self.register_buffer("uncond_condition_embedding", nn.Parameter(torch.zeros(1024, 1280)))
        # self.condition_feature_proj = nn.Linear(1280, 384)    

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)
        # nn.init.constant_(self.qwenvl_mlp.cap_proj.fc1.weight, 0)
        # nn.init.constant_(self.qwenvl_mlp.cap_proj.fc2.weight, 0)

        
        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def condition_dropout(self, x, p=0.5, training=True):
        if not training or p == 0:
            return x
        keep_p = 1 - p
        # 生成 (B, L, 1) 的 mask
        mask = torch.rand(x.size(0), x.size(1), 1, device=x.device) < keep_p
        mask = mask.float() / keep_p  # scale
        return x * mask
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)  # 
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)


    
    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        qwenvl_embedding: Optional[torch.Tensor] = None,
        source_image: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):  
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings,drop_ids = self.cls_embedding(cond_idx, train=self.training)
            cond_embeddings = cond_embeddings[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            # self.target_embedding = self.tok_embeddings(targets)
            # if condition is not None:
            if self.encoder_name == 'dino':
                condition_embeddings = self.adapter(condition)
                condition_embeddings = self.adapter_mlp(condition_embeddings)
            elif self.encoder_name == 'vqgan':
                condition_embeddings = self.tok_embeddings(condition)
            elif self.encoder_name == 'qwen':
                if self.use_adapter:
                    source_image_embedding = self.adapter(source_image)
                    if self.use_fusion:
                        qwenvl_embedding = self.fusionblock(source_image_embedding, qwenvl_embedding, padding_mask=padding_mask)
                        condition_embeddings = qwenvl_embedding
                    else:
                        if self.qwenvl_image_size == 512:
                            B, L, D = qwenvl_embedding.shape
                            qwenvl_embedding = qwenvl_embedding.view(B, 16, 16, D).permute(0,3,1,2)
                            qwenvl_embedding = F.interpolate(qwenvl_embedding,size=(32,32),mode='bilinear',align_corners=False).permute(0,2,3,1).flatten(1,2)
                
                
                
                        
                        condition_embeddings = self.qwenvl_mlp(source_image_embedding)
                        
                        
                if self.use_connector:
                    condition_embeddings = self.qwenvl_connector(condition_embeddings)
            self.condition_token = condition_embeddings
            if self.use_prefilling:
                prefilling_embedding = self.prefilling_mlp(self.prefilling_adapter(source_image))
                cond_embeddings[:, :len(prefilling_embedding[0])] = cond_embeddings[:, :len(prefilling_embedding[0])] + prefilling_embedding
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
            # mask = None
            
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)
                token_embeddings = token_embeddings[:,:self.cls_token_num]
                if self.encoder_name == 'dino':
                    condition_embeddings = self.condition_mlp(condition, train=self.training)
                    self.condition_token = condition_embeddings
                elif self.encoder_name == 'vqgan':
                    condition_embeddings = self.tok_embeddings(condition)
                    self.condition_token = condition_embeddings
                elif self.encoder_name == 'qwen':
                    
                    
                    
                    
                    
                    condition_embeddings = qwenvl_embedding
                    if self.use_connector:
                        condition_embeddings = self.qwenvl_connector(condition_embeddings)
                    
                    self.condition_token = condition_embeddings * self.control_strength
                    if self.use_prefilling:
                        print('use_prefilling')
                        prefilling_embedding = self.prefilling_mlp(self.prefilling_adapter(source_image))
                        token_embeddings[:, :len(prefilling_embedding[0])] = token_embeddings[:, :len(prefilling_embedding[0])] + prefilling_embedding
                    

            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            # h = token_embeddings
            self.freqs_cis = self.freqs_cis

        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        
        for i, layer in enumerate(self.layers):
            if self.training:
                h[:, self.cls_token_num-1:] = h[:, self.cls_token_num-1:] + self.condition_token * self.control_strength
                
            else:
                if len(input_pos)>1:
                    h[:,-1:] = h[:, -1:] + self.condition_token[:,0:1]
                else:
                    h = h + self.condition_token[:,input_pos-self.cls_token_num+1]
            h = layer(h, freqs_cis, input_pos, mask)
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            logits = logits[:, self.cls_token_num - 1:].contiguous()
        # if we are given some desired targets also calculate the loss
        loss = None
        # if valid is not None:
        #     loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
        #     valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
        #     loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        # elif targets is not None:
        if self.training:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
            
            
            
            
            return logits, loss, None
        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}
