import torch
from typing import Tuple, Optional


class GraphEncoder:
    """Utility for graph construction from point clouds."""
    
    @staticmethod
    def distance_to_weight(distances: torch.Tensor, kernel: str = "gaussian", sigma: float = 1.0) -> torch.Tensor:
        """
        Convert distances to edge weights based on kernel type.
        
        Args:
            distances: Edge distances [E]
            kernel: Type of kernel ("gaussian", "inverse", "linear")
            sigma: Kernel bandwidth parameter (for gaussian kernel)
            
        Returns:
            edge_weights: Computed edge weights [E]
        """
        
        if kernel == "gaussian":
            return torch.exp(-distances**2 / (2 * sigma**2))
        elif kernel == "inverse":
            return 1.0 / (1.0 + distances)
        elif kernel == "linear":
            max_dist = distances.max()
            if max_dist > 0:
                return 1.0 - (distances / max_dist)
            else:
                return torch.ones_like(distances)
        else:
            raise ValueError(f"Unknown distance kernel: {kernel}")
    
    
    @staticmethod
    def create_fully_connected_edges(batch_assignment: torch.Tensor) -> torch.Tensor:
        """
        Create fully connected edge structure for batched graphs.
        
        Args:
            batch_assignment: Batch assignment for each node [N]
            
        Returns:
            edge_index: Edge connectivity [2, E]
        """
        
        device = batch_assignment.device
        num_graphs = batch_assignment.max().item() + 1
        edge_list = []
        
        for b in range(num_graphs):
            node_mask = (batch_assignment == b)
            node_indices = torch.where(node_mask)[0]
            n_nodes = len(node_indices)
            
            if n_nodes > 1:
                src = node_indices.repeat_interleave(n_nodes)
                dst = node_indices.repeat(n_nodes)
                # Remove self-loops
                mask = src != dst
                edges = torch.stack([src[mask], dst[mask]], dim=0)
                edge_list.append(edges)
        
        if edge_list:
            return torch.cat(edge_list, dim=1)
        else:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    
    @staticmethod
    def create_edges_with_attributes(coords: torch.Tensor,  batch_assignment: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create fully connected graphs with edge attributes.
        
        Args:
            coords: Node coordinates [N, 3] are spatial
            batch_assignment: Optional batch assignment for batched graphs [N]
            
        Returns:
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge attributes [E, 4] containing [distance, dx, dy, dz]
        """
        
        if coords.shape[1] != 3:
            raise ValueError("Coordinates must be spatial dimensions [N, 3]")
        
        if batch_assignment is not None:
            edge_index = GraphEncoder.create_fully_connected_edges(batch_assignment)
        else:
            n_nodes = coords.shape[0]
            if n_nodes > 1:
                src = torch.arange(n_nodes, device=coords.device).repeat_interleave(n_nodes)
                dst = torch.arange(n_nodes, device=coords.device).repeat(n_nodes)
                mask = src != dst
                edge_index = torch.stack([src[mask], dst[mask]], dim=0)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=coords.device)
        
        if edge_index.shape[1] > 0:
            src_coords = coords[edge_index[0]]
            dst_coords = coords[edge_index[1]]
            
            direction = dst_coords - src_coords
            distance = torch.norm(direction, dim=1, keepdim=True)
            
            edge_attr = torch.cat([distance, direction], dim=1)
        else:
            edge_attr = torch.zeros((0, 4), device=coords.device) # TODO: check validity of this
        
        return edge_index, edge_attr
