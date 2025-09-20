import torch.nn as nn

from config.config import ModelConfig
from model.hybrid_attention import HybridAttention
from model.pe import DegreeScaler

class TransformerBlock(nn.Module):
    """Transformer block with Latent Attention and configurable normalization"""
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm_style = config.norm_style
        
        # Hybrid attention (GRIT or MLA based on layer)
        is_last_layer = (layer_idx == config.num_layers - 1)
        
        self.attention = HybridAttention(
            hidden_dim = config.hidden_dim,
            num_heads = config.num_heads,
            layer_idx = layer_idx,
            use_mla_layers = config.use_mla_layers,
            dropout = config.dropout,
            attn_dropout = config.attn_dropout,
            edge_enhance = config.update_edge_repr,
            kv_latent_dim = config.mla_kv_latent_dim,
            edge_correction_dim_ratio = config.edge_correction_dim_ratio,
            initial_edge_weight = config.initial_edge_weight,
            learnable_edge_weight = config.learnable_edge_weight,
            is_last_layer = is_last_layer
        )
        
        # Degree scaler # TODO deprecate
        if config.use_degree_scaler:
            self.degree_scaler = DegreeScaler(config.hidden_dim)
        
        # Normalization layers based on style
        if self.norm_style in ["pre", "pre_post"]:
            # Pre-normalization layers
            self.norm1_pre = nn.RMSNorm(config.hidden_dim)
            self.norm2_pre = nn.RMSNorm(config.hidden_dim)
        
        if self.norm_style in ["post", "pre_post"]:
            # Post-normalization layers
            self.norm1_post = nn.RMSNorm(config.hidden_dim)
            self.norm2_post = nn.RMSNorm(config.hidden_dim)

        # Dense FFN
        ffn_hidden_dim = int(config.hidden_dim * config.ffn_dim_multiplier)
        
        # SwiGLU activation
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_hidden_dim * 2),  # Gate and up projection
            SwiGLU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_hidden_dim, config.hidden_dim),  # Down projection
            nn.Dropout(config.dropout)
        )


    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Attention block with normalization
        if self.norm_style in ["pre", "pre_post"]:
            # Pre-normalization
            x_normed = self.norm1_pre(x)
            x_attn, edge_attr_new = self.attention(x_normed, edge_index, edge_attr, batch)
        else:
            # No pre-norm (original behavior)
            x_attn, edge_attr_new = self.attention(x, edge_index, edge_attr, batch)
        
        # Residual connection
        x = x + x_attn
        
        # Apply degree scaling if enabled # TODO Deprecate
        if self.config.use_degree_scaler:
            x = self.degree_scaler(x, edge_index)
        
        # Post-normalization after attention (if enabled)
        if self.norm_style in ["post", "pre_post"]:
            x = self.norm1_post(x)
        
        # FFN block with normalization
        if self.norm_style in ["pre", "pre_post"]:
            # Pre-normalization
            x_normed = self.norm2_pre(x)
            x_ffn = self.ffn(x_normed)
        else:
            # No pre-norm (original behavior)
            x_ffn = self.ffn(x)
        
        # Residual connection
        x = x + x_ffn
        
        # Post-normalization after FFN (if enabled)
        if self.norm_style in ["post", "pre_post"]:
            x = self.norm2_post(x)
        
        # Update edge attributes
        if edge_attr_new is not None:
            edge_attr = edge_attr_new
        
        return x, edge_attr


class SwiGLU(nn.Module):
    """SwiGLU: SiLU(x) * gate"""
    
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return nn.functional.silu(x) * gate
