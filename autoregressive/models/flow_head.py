import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Modulate tensor with shift and scale.
    
    Args:
        x: Input tensor
        shift: Shift tensor (optional)
        scale: Scale tensor (optional)
        
    Returns:
        Modulated tensor
    """
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        Initialize TimestepEmbedder.
        
        Args:
            hidden_size: Hidden layer size
            frequency_embedding_size: Frequency embedding size
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: 1-D tensor of N indices, one per batch element (may be fractional)
            dim: Output dimension
            max_period: Controls the minimum frequency of the embeddings
            
        Returns:
            (N, D) tensor of positional embeddings
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: Input tensor
            
        Returns:
            Embedded tensor
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

class ResBlock(nn.Module):
    """Residual block with adaptive layer normalization."""
    
    def __init__(self, channels: int, mlp_ratio: float = 1.0):
        """
        Initialize ResBlock.
        
        Args:
            channels: Number of channels
            mlp_ratio: MLP ratio for intermediate layer size
        """
        super().__init__()
        self.channels = channels
        self.intermediate_size = int(channels * mlp_ratio)

        self.in_ln = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.channels),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            y: Conditioning tensor
            
        Returns:
            Output tensor
        """
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """Final layer with adaptive layer normalization."""
    
    def __init__(self, model_channels: int, out_channels: int):
        """
        Initialize FinalLayer.
        
        Args:
            model_channels: Model channels
            out_channels: Output channels
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            c: Conditioning tensor
            
        Returns:
            Output tensor
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    """Simple MLP with adaptive layer normalization."""
    
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        dim: int = 1536,
        layers: int = 12,
        mlp_ratio: float = 1.0
    ):
        """
        Initialize SimpleMLPAdaLN.
        
        Args:
            input_dim: Input dimension
            cond_dim: Conditioning dimension
            dim: Hidden dimension
            layers: Number of layers
            mlp_ratio: MLP ratio
        """
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.dim = dim
        self.layers = layers
        self.mlp_ratio = mlp_ratio
        
        self.time_embed = TimestepEmbedder(dim)
        self.cond_embed = nn.Linear(cond_dim, dim)
        self.input_proj = nn.Linear(input_dim, dim)

        res_blocks = []
        for _ in range(layers):
            res_blocks.append(ResBlock(dim, mlp_ratio))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.final_layer = FinalLayer(dim, input_dim)

        self.grad_checkpointing = False

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize model weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape (batch_size, input_dim)
            t: Time tensor with shape (batch_size,)
            c: Conditioning tensor with shape (batch_size, cond_dim)
            
        Returns:
            Output tensor with shape (batch_size, input_dim)
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        for i, block in enumerate(self.res_blocks):
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, y, use_reentrant=True)
            else:
                x = block(x, y)

        output = self.final_layer(x, y)
        
        return output

def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape time tensor t to broadcastable dimension of x.
    
    Args:
        t: Time vector with shape [batch_size]
        x: Data point with shape [batch_size, ...]
        
    Returns:
        Reshaped time tensor
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def randn_tensor(shape: tuple, noise_repeat: int, device: torch.device) -> torch.Tensor:
    """
    Generate random tensor with noise repeat.
    
    Args:
        shape: Target shape (actual sample count: batch_size * seq, dim)
        noise_repeat: Noise repeat factor
        device: Target device
        
    Returns:
        Random tensor
    """
    # Input shape is real sample count (bsz*seq, dim), bsz*seq is entire image token, all randomized
    return torch.randn(shape, device=device)


def time_shift_fn(t: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    """
    Time redistribution function (biased towards t≈0 noise end).
    
    Args:
        t: Input time tensor
        shift: Shift parameter (shift>=1 biases towards small t)
        
    Returns:
        Shifted time tensor
    """
    # Equivalent to t' = t / (t + (1 - t) * shift), shift>=1 biases towards small t
    return t / (t + (1.0 - t) * shift)


class FlowMatchingHead(nn.Module):
    """Flow matching head for generative modeling."""
    
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        dim: int = 1536,
        layers: int = 12,
        mlp_ratio: float = 1.0,
        sample_t: str = "logit-normal",
        timeshift: Optional[float] = None,
    ):
        """
        Initialize FlowMatchingHead.
        
        Args:
            input_dim: Input dimension
            cond_dim: Conditioning dimension
            dim: Hidden dimension
            layers: Number of layers
            mlp_ratio: MLP ratio
            sample_t: Time sampling method ("logit-normal")
            timeshift: Time shift parameter
        """
        super(FlowMatchingHead, self).__init__()
        self.input_dim = input_dim
        self.timeshift = timeshift 
        self.sample_t = sample_t
        print(f"[FlowMatchingHead]: timeshift: {timeshift}")
        self.net = SimpleMLPAdaLN(
            input_dim=input_dim, 
            cond_dim=cond_dim, 
            dim=dim, 
            layers=layers, 
            mlp_ratio=mlp_ratio,
        )

    @property
    def dtype(self) -> torch.dtype:
        """Get model data type."""
        return self.net.input_proj.weight.dtype

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return self.net.input_proj.weight.device

    @property
    def trainable_params(self) -> int:
        """Get number of trainable parameters."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    def forward(
        self,
        target: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        keep_dim: bool = False,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            target: Target tensor with shape (batch_size, input_dim)
            c: Conditioning tensor with shape (batch_size, cond_dim)
            mask: Mask tensor with shape (batch_size,) (optional)
            keep_dim: Whether to keep non-batch dimensions (default False)
            
        Returns:
            Computed loss
        """
        noise = torch.randn_like(target)

        if self.sample_t == "logit-normal":
            # sample t from logit-normal distribution
            u = torch.normal(mean=0.0, std=1.0, size=(len(target),))
            t = (1 / (1 + torch.exp(-u))).to(target)

            # Apply time bias
            if self.timeshift is not None and self.timeshift < 1.0:
                t = time_shift_fn(t, self.timeshift)
        else:
            assert False, "Invalid sample_t, must [logit-normal]"

        # linear interpolation between target and noise
        xt = expand_t(t, target) * target + (1 - expand_t(t, target)) * noise
        ut = target - noise

        model_output = self.net(xt, t, c)
            
        loss = (model_output.float() - ut.float()) ** 2
        if not keep_dim:
            # Default behavior: average over all non-batch dimensions
            loss = torch.mean(loss, dim=list(range(1, len(loss.size()))))

        if mask is not None:
            # Weighted average by mask
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean() if not keep_dim else loss

        return loss
    
    def get_score_from_velocity(
        self, 
        velocity: torch.Tensor, 
        x: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Wrapper function: transform velocity prediction model to score.
        
        Args:
            velocity: Velocity model output with shape [batch_size, ...]
            x: x_t data point with shape [batch_size, ...]
            t: Time tensor with shape [batch_size,]
            
        Returns:
            Score tensor
        """
        t = expand_t(t, x)
        alpha_t, d_alpha_t = t, 1
        sigma_t, d_sigma_t = 1 - t, -1
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_velocity_from_cfg(
        self, 
        velocity: torch.Tensor, 
        cfg: float, 
        cfg2: float, 
        cfg_mult: int
    ) -> torch.Tensor:
        """
        Get velocity from CFG (Classifier-Free Guidance).
        
        Args:
            velocity: Input velocity
            cfg: CFG scale
            cfg2: Second CFG scale
            cfg_mult: CFG multiplier
            
        Returns:
            CFG-adjusted velocity
        """
        if cfg_mult == 2:
            cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
            velocity = uncond_v + cfg * (cond_v - uncond_v)
        elif cfg_mult == 3:
            cond_v, uncond_v1, uncond_v2 = torch.chunk(velocity, 3, dim=0)
            velocity = uncond_v2 + cfg2 * (uncond_v1 - uncond_v2) + cfg * (cond_v - uncond_v1)
        return velocity

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        cfg: float = 1.0,
        cfg2: float = 1.0,
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        last_step_size: float = 0.04,
        noise_repeat: int = 1,
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the model.
        
        Args:
            c: Conditioning tensor with shape (batch_size, cond_dim)
            cfg: CFG scale
            cfg2: Second CFG scale
            timesteps_shift: Timesteps shift
            num_sampling_steps: Number of sampling steps
            last_step_size: Size of last step
            noise_repeat: Noise repeat factor
            progress: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        cfg_mult = 1
        if cfg > 1.0:
            cfg_mult += 1
        if cfg2 > 1.0:
            cfg_mult += 1

        noise = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)

        mean_x = noise
        x = noise
        xs = []
        t0, t1 = 0, 1 - last_step_size
        timesteps = torch.linspace(t0, t1, num_sampling_steps)
        timestep_pairs = zip(timesteps[:-1], timesteps[1:])
        if progress:
            from tqdm.auto import tqdm
            # Calculate total iterations (timesteps length minus 1)
            total = len(timesteps) - 1 if hasattr(timesteps, '__len__') else None
            timestep_pairs = tqdm(timestep_pairs, total=total, desc="Processing flow matching")
            
        for ti, tj in timestep_pairs:
            dt = tj - ti

            combined = torch.cat([x] * cfg_mult, dim=0)
            velocity = self.net(combined.to(c.dtype), ti.expand(c.shape[0]).to(c), c)
            velocity = velocity.to(torch.float32)

            velocity = self.get_velocity_from_cfg(velocity, cfg, cfg2, cfg_mult)
            score = self.get_score_from_velocity(velocity, x, ti.expand(x.shape[0]).to(x))
            drift = velocity + (1 - expand_t(ti.expand(x.shape[0]).to(x), x)) * score
            w_cur = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)
            dw = w_cur * torch.sqrt(dt)

            mean_x = x + drift * dt
            x = mean_x + torch.sqrt(2 * (1 - expand_t(ti.expand(x.shape[0]).to(x), x))) * dw
            xs.append(x)

        combined = torch.cat([xs[-1]] * cfg_mult, dim=0)
        velocity = self.net(combined.to(c.dtype), timesteps[-1].expand(c.shape[0]).to(c), c)
        velocity = velocity.to(torch.float32)

        velocity = self.get_velocity_from_cfg(velocity, cfg, cfg2, cfg_mult)
        x = xs[-1] + velocity * last_step_size
        xs.append(x)

        if len(xs) != num_sampling_steps:
            raise ValueError(f"Samples ({len(xs)}) does not match the number of steps ({num_sampling_steps})")

        return xs[-1].to(c.dtype)