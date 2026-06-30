import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticFusionBlock(nn.Module):
    def __init__(self, dim, query_dim=384, seq_len=1024, n_heads=8, expansion_ratio=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_ratio, dim),
            nn.Dropout(dropout)
        )
        
        self.gate = nn.Parameter(torch.zeros(1))  
        
        self.a_proj = nn.Sequential(
            nn.Linear(2048, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.b_proj = nn.Sequential(
            nn.Linear(query_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.pos_bias = nn.Parameter(torch.zeros(seq_len, dim))  
        

        self.ada_norm_weight = nn.Linear(dim, dim)
        self.ada_norm_bias = nn.Linear(dim, dim)

    def forward(self, B, A, padding_mask=None):

        B = self.b_proj(B)
        
        B = B + self.pos_bias[:B.size(1)]  
        
        residual = B
        

        A = self.a_proj(A)  
        
        # Step 2: Cross-Attention: B attends to A_enhanced
        q = self.norm1(B)
        k = A
        v = A

        out_attn, _ = self.attn(q, k, v, key_padding_mask=padding_mask)
        

        B = residual + self.gate.tanh() * out_attn  # tanh
        
        # Step 3: AdaLN + FFN
        norm_weight = self.ada_norm_weight(A.mean(1)).unsqueeze(1)  # (B, 1, D)
        norm_bias = self.ada_norm_bias(A.mean(1)).unsqueeze(1)
        
        B = B * (1 + norm_weight) + norm_bias  
        
        # FFN
        B = B + self.ffn(self.norm2(B))
        
        return B
    
    

if __name__ == '__main__':
    import torch
    fusionblock = SemanticFusionBlock(dim=1280)
    print(f"fusionblock Parameters: {sum(p.numel() for p in fusionblock.parameters()):,}")
    A = torch.randn(16,256+30,2048)
    B = torch.randn(16, 1024, 384)
    output = fusionblock(B, A)
    print(output.shape)