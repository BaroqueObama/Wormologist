import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiScaleRRWPFiltration(nn.Module):
    """Multi-scale Random Walk Filtration: concatenation: [coords || RRWP_scale_1 || RRWP_scale_2 || ... || RRWP_scale_N]"""
    
    def __init__(self, num_scales: int = 4, num_steps: int = 31, min_sigma: float = 0.1, max_sigma: float = 3.0, learnable_scales: bool = True):
        super().__init__()
        
        self.num_scales = num_scales
        self.num_steps = num_steps
        
        # Initialize log-spaced sigmas
        log_sigmas = torch.linspace(
            torch.log(torch.tensor(min_sigma)),
            torch.log(torch.tensor(max_sigma)),
            num_scales
        )
        
        if learnable_scales:
            self.log_sigmas = nn.Parameter(log_sigmas)
        else:
            self.register_buffer('log_sigmas', log_sigmas)
    
    
    def compute_multiscale_rrwp(self, coords: torch.Tensor, edge_index: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-scale RRWP features from point cloud coordinates.
        
        Args:
            coords: Node coordinates [N, 3]
            edge_index: Optional edge connectivity for output edge features [2, E]
            batch: Optional batch assignment for batched graphs [N]
        
        Returns:
            node_features: Concatenated multi-scale node RRWP features [N, num_scales * num_steps]
            edge_features: Concatenated multi-scale edge RRWP features [E, num_scales * num_steps] or None
        """
        
        N = coords.shape[0]
        device = coords.device
        
        dist_matrix = torch.cdist(coords, coords)  # [N, N]
        
        all_node_features = []
        all_edge_features = []
        
        # Get sorted sigmas for stable training # TODO: Hopefully this doesn't mess with the gradients
        sigmas = torch.exp(self.log_sigmas).sort()[0]
        
        for sigma in sigmas:
            weights = torch.exp(-dist_matrix**2 / (2 * sigma**2))
            # Remove self-loops
            weights = weights - torch.diag(weights.diagonal())
            
            # Row-normalize to get transition matrix
            P = F.normalize(weights, p=1, dim=1)
            P = torch.nan_to_num(P, 0.0) # isolated nodes
            
            P_k = torch.eye(N, device=device)
            node_feats = []
            edge_feats_full = []
            
            for k in range(self.num_steps):
                P_k = P_k @ P
                node_feats.append(P_k.diagonal())
                edge_feats_full.append(P_k)
            
            scale_node_feats = torch.stack(node_feats, dim=-1)  # [N, num_steps]
            scale_edge_matrix = torch.stack(edge_feats_full, dim=-1)  # [N, N, num_steps]
            
            all_node_features.append(scale_node_feats)
            
            if edge_index is not None:
                scale_edge_feats = scale_edge_matrix[edge_index[0], edge_index[1]]  # [E, num_steps]
                all_edge_features.append(scale_edge_feats)
        
        node_features = torch.cat(all_node_features, dim=-1)  # [N, num_scales * num_steps]
        
        edge_features = None
        if edge_index is not None and len(all_edge_features) > 0:
            edge_features = torch.cat(all_edge_features, dim=-1)  # [E, num_scales * num_steps]
        
        return node_features, edge_features
