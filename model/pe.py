import torch
import torch.nn as nn


class RRWPEncoder(nn.Module):
    """Relative Random Walk Probabilities encoder"""
    
    def __init__(self, k_steps: int):
        super().__init__()
        self.k_steps = k_steps
    
    
    def forward(self, edge_index, num_nodes, edge_weight=None):
        device = edge_index.device
        
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        if edge_weight is not None:
            adj[edge_index[0], edge_index[1]] = edge_weight
        else:
            adj[edge_index[0], edge_index[1]] = 1.0
        
        deg = adj.sum(dim=1)
        deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        M = deg_inv.unsqueeze(1) * adj
        
        rrwp_features = []
        M_power = torch.eye(num_nodes, device=device)
        
        for _ in range(self.k_steps):
            rrwp_features.append(M_power)
            M_power = torch.mm(M_power, M)
        
        rrwp = torch.stack(rrwp_features, dim=-1)
        node_pe = rrwp.diagonal(dim1=0, dim2=1).t()
        edge_pe = rrwp[edge_index[0], edge_index[1]]
        
        return node_pe, edge_pe


class PEModule(nn.Module):
    """PE module"""
    
    def __init__(self, k_hops: int):
        super().__init__()
        self.k_hops = k_hops
        
        self.node_pe_proj = None
        self.edge_pe_proj = None
        
        self.pe_encoder = RRWPEncoder(k_hops)
    
    
    def forward(self, edge_index, num_nodes, edge_weight=None):
        node_pe, edge_pe = self.pe_encoder(edge_index, num_nodes, edge_weight)
        return node_pe, edge_pe


# TODO: Not used, consider deprecating
class DegreeScaler(nn.Module):
    """Inject degree information into node representations"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.theta1 = nn.Parameter(torch.ones(hidden_dim))
        self.theta2 = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_scale = torch.log(1 + deg).unsqueeze(1)
        x_out = x * self.theta1 + deg_scale * x * self.theta2
        return x_out
