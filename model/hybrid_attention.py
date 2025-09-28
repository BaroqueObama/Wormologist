"""
GRIT Attention taken from: https://github.com/LiamMa/GRIT which is the official implementation of 
Graph Inductive Biases in Transformers without Message Passing (Ma et al., ICML 2023) [https://arxiv.org/abs/2305.17589]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter
from einops import rearrange


class HybridAttention(nn.Module):
    """Either standard GRIT attention or MLA with learnable edge correction"""
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        layer_idx: int,
        use_mla_layers: list = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0, # TODO: Deprecate Attention Dropout
        edge_enhance: bool = True,
        kv_latent_dim: int = 64,
        edge_correction_dim_ratio: float = 0.25,
        initial_edge_weight: float = 0.1,
        learnable_edge_weight: bool = True,
        is_last_layer: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_enhance = edge_enhance
        self.is_last_layer = is_last_layer
        self.layer_idx = layer_idx
        
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        
        self.use_mla = use_mla_layers and layer_idx in use_mla_layers
        
        if self.use_mla:
            self._setup_mla_hybrid(hidden_dim, num_heads, kv_latent_dim, edge_correction_dim_ratio, initial_edge_weight, learnable_edge_weight, attn_dropout)
        else:
            self._setup_grit_attention(hidden_dim, num_heads, edge_enhance)
        
        # Common components
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        
        if edge_enhance and not is_last_layer:
            self.W_Eo = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.act = nn.ReLU()

    def _setup_mla_hybrid(self, hidden_dim, num_heads, kv_latent_dim, edge_correction_dim_ratio, initial_edge_weight, learnable_edge_weight, attn_dropout):
        """Setup MLA with edge correction components"""

        self.mla_scale = (hidden_dim // num_heads) ** -0.5
        self.mla_W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # KV joint compression: hidden_dim -> kv_latent_dim
        self.mla_W_DKV = nn.Linear(hidden_dim, kv_latent_dim, bias=False)
        
        # KV decompression: kv_latent_dim -> full dimension for K and V
        self.mla_W_UK = nn.Linear(kv_latent_dim, hidden_dim, bias=False)
        self.mla_W_UV = nn.Linear(kv_latent_dim, hidden_dim, bias=False)
        
        # Normalization for stability
        self.mla_kv_norm = nn.RMSNorm(kv_latent_dim)
        
        # Lightweight edge correction components # TODO: bigger for edge may be better
        edge_dim = int(hidden_dim * edge_correction_dim_ratio)
        self.edge_dim = edge_dim
        self.edge_heads = max(1, num_heads // 4)  # Fewer heads for edge attention
        self.edge_head_dim = edge_dim // self.edge_heads
        
        # Edge attention projections
        self.edge_q = nn.Linear(hidden_dim, edge_dim)
        self.edge_k = nn.Linear(hidden_dim, edge_dim)
        self.edge_v = nn.Linear(hidden_dim, edge_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # Edge attribute processing
        self.edge_w = nn.Linear(hidden_dim, edge_dim)
        self.edge_b = nn.Linear(hidden_dim, edge_dim)
        self.edge_a = nn.Linear(edge_dim, self.edge_heads)
        
        # Learnable weight for edge correction
        if learnable_edge_weight:
            self.edge_weight = nn.Parameter(torch.tensor(initial_edge_weight))
        else:
            self.register_buffer('edge_weight', torch.tensor(initial_edge_weight))
    
    def _setup_grit_attention(self, hidden_dim, num_heads, edge_enhance):
        """Setup standard GRIT attention components"""
        
        # Standard GRIT attention components
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_Ew = nn.Linear(hidden_dim, hidden_dim)
        self.W_Eb = nn.Linear(hidden_dim, hidden_dim)
        self.W_A = nn.Linear(hidden_dim, num_heads)
        
        if edge_enhance:
            self.W_Ev = nn.Linear(hidden_dim, hidden_dim)
    
    def signed_sqrt(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8)
    
    def compute_edge_correction(self, x, edge_index, edge_attr=None):
        """Compute lightweight edge-based attention correction"""
        
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return torch.zeros_like(x)
        
        q = self.edge_q(x).view(num_nodes, self.edge_heads, self.edge_head_dim)
        k = self.edge_k(x).view(num_nodes, self.edge_heads, self.edge_head_dim)
        v = self.edge_v(x).view(num_nodes, self.edge_heads, self.edge_head_dim)
        
        src, tgt = edge_index
        q_i = q[src]  # [num_edges, edge_heads, edge_head_dim]
        k_j = k[tgt]
        
        # Compute attention features
        attn_features = q_i + k_j
        
        # Apply edge attributes if available
        if edge_attr is not None:
            e_w = self.edge_w(edge_attr).view(num_edges, self.edge_heads, self.edge_head_dim)
            e_b = self.edge_b(edge_attr).view(num_edges, self.edge_heads, self.edge_head_dim)
            attn_features = self.signed_sqrt(attn_features * e_w) + e_b
        
        attn_features = self.act(attn_features)
        
        # Compute attention scores
        attn_scores = self.edge_a(attn_features.view(num_edges, -1))
        attn_scores = pyg_softmax(attn_scores, tgt, num_nodes=num_nodes)
        attn_scores = self.attn_dropout(attn_scores)
        attn_scores = attn_scores.view(num_edges, self.edge_heads, 1)
        
        # Apply attention to values
        v_j = v[src]
        messages = v_j * attn_scores
        
        # Aggregate messages
        out = torch.zeros(num_nodes, self.edge_heads, self.edge_head_dim, device=x.device)
        scatter(messages, tgt, dim=0, out=out, reduce='add')
        
        # Project back to hidden dimension
        out = out.view(num_nodes, self.edge_dim)
        out = self.edge_proj(out)
        
        return out
    
    def apply_mla(self, x, mask=None):
        """Apply MLA attention mechanism, expects batched input [B, N, D]"""
        
        assert x.dim() == 3, f"MLA expects 3D input [batch, nodes, dim], got {x.dim()}D with shape {x.shape}"
        
        batch_size, seq_len, hidden_dim = x.shape
        
        Q = self.mla_W_Q(x)
        
        c_KV = self.mla_W_DKV(x)
        c_KV = self.mla_kv_norm(c_KV)
        
        K = self.mla_W_UK(c_KV)
        V = self.mla_W_UV(c_KV)
        
        # Reshape for multi-head attention
        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.mla_scale
        
        # Apply mask if provided
        if mask is not None and mask.any():
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
                mask = mask.expand(-1, self.num_heads, seq_len, -1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        
        # Always return 3D tensor
        return attn_output
    
    def forward_mla_hybrid(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass for MLA with edge correction"""
        
        assert batch is not None, ("Batch tensor is required for MLA attention.")
        
        # Process batched graphs for MLA
        batch_size = batch.max().item() + 1
        x_batched = []
        for b in range(batch_size):
            mask = (batch == b)
            x_batched.append(x[mask])
        
        # Pad sequences for batch processing
        max_len = max(x_b.size(0) for x_b in x_batched)
        x_padded = torch.zeros(batch_size, max_len, x.size(1), device=x.device)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x.device)
        
        for b, x_b in enumerate(x_batched):
            len_b = x_b.size(0)
            x_padded[b, :len_b] = x_b
            masks[b, :len_b] = True
        
        # Apply MLA (expects 3D input)
        x_global = self.apply_mla(x_padded, masks)
        
        # Unpad
        x_global_list = []
        for b in range(batch_size):
            mask = (batch == b)
            num_nodes_b = mask.sum().item()
            x_global_list.append(x_global[b, :num_nodes_b])
        x_global = torch.cat(x_global_list, dim=0)
        
        # Compute edge correction
        x_edge = self.compute_edge_correction(x, edge_index, edge_attr)
        
        # Ensure dimensions match for addition
        assert x_global.shape == x_edge.shape, \
            f"Shape mismatch: x_global {x_global.shape} vs x_edge {x_edge.shape}"
        
        # Combine with learnable weight
        x_out = x_global + self.edge_weight * x_edge
        
        # Final projection
        x_out = self.W_O(x_out)
        x_out = self.dropout(x_out)
        
        return x_out
    
    def forward_grit(self, x, edge_index, edge_attr=None):
        """Forward pass for standard GRIT attention"""
        
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        Q = self.W_Q(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.W_K(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.W_V(x).view(num_nodes, self.num_heads, self.head_dim)
        
        src, tgt = edge_index
        Q_i = Q[src]
        K_j = K[tgt]
        
        attn_features = Q_i + K_j
        
        if edge_attr is not None:
            E_w = self.W_Ew(edge_attr).view(num_edges, self.num_heads, self.head_dim)
            E_b = self.W_Eb(edge_attr).view(num_edges, self.num_heads, self.head_dim)
            attn_features = self.signed_sqrt(attn_features * E_w) + E_b
        
        attn_features = self.act(attn_features)
        attn_scores = self.W_A(attn_features.view(num_edges, -1))
        attn_scores = pyg_softmax(attn_scores, tgt, num_nodes=num_nodes)
        attn_scores = self.attn_dropout(attn_scores)
        attn_scores_out = attn_scores.view(num_edges, self.num_heads, 1)
        
        V_j = V[src]
        messages = V_j * attn_scores_out
        
        if self.edge_enhance and edge_attr is not None:
            E_v = self.W_Ev(edge_attr).view(num_edges, self.num_heads, self.head_dim)
            messages = messages + E_v * attn_scores_out
        
        x_out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        scatter(messages, tgt, dim=0, out=x_out, reduce='add')
        
        x_out = x_out.view(num_nodes, self.hidden_dim)
        x_out = self.W_O(x_out)
        x_out = self.dropout(x_out)
        
        return x_out, attn_features
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Main forward pass"""
        
        assert batch is not None, ("Batch tensor is required for attention forward pass.")
        
        if self.use_mla:
            # MLA with edge correction
            x_out = self.forward_mla_hybrid(x, edge_index, edge_attr, batch)
            
            edge_attr_out = None
            if self.edge_enhance and edge_attr is not None and not self.is_last_layer:
                edge_attr_out = self.W_Eo(edge_attr)
                edge_attr_out = self.dropout(edge_attr_out)
            
        else:
            # Standard GRIT (batch not used in computation but required for consistency)
            x_out, attn_features = self.forward_grit(x, edge_index, edge_attr)
            
            edge_attr_out = None
            if self.edge_enhance and edge_attr is not None and not self.is_last_layer:
                edge_attr_out = self.W_Eo(attn_features.view(edge_index.size(1), -1))
                edge_attr_out = self.dropout(edge_attr_out)
        
        return x_out, edge_attr_out
