import torch
import torch.nn as nn
from typing import Optional, Dict

from model.transformer_block import TransformerBlock, SwiGLU
from config.config import ModelConfig


class Wormologist(nn.Module):
    """GRIT model with Sinkhorn matching"""
    def __init__(self, config: ModelConfig, node_input_dim: int, edge_input_dim: int):
        super().__init__()
        self.config = config
        
        self.node_encoder = nn.Linear(node_input_dim, config.hidden_dim) # Features already include PE
        
        # Edge encoder for concatenated edge features + PE
        if edge_input_dim > 0:
            self.edge_encoder = nn.Linear(edge_input_dim, config.hidden_dim)
        else:
            self.edge_encoder = None
        
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)])
        
        # Classification head for canonical ID prediction
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            SwiGLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.out_dim)
        )

        if config.cell_type_out_dim and config.cell_type_out_dim > 0:
            self.cell_type_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                SwiGLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.cell_type_out_dim)
            )
        else:
            self.cell_type_head = None
    
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, batch: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features with PE concatenated [num_nodes, node_input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features with PE [num_edges, edge_input_dim]
            batch: Batch assignment for each node [num_nodes]
        
        Returns:
            Dictionary:
             - logits: Classification logits [batch_size, max_nodes, 558]
        """
        
        # Batch tensor is required - fail loudly if missing
        assert batch is not None, ("Batch tensor is required for model forward pass.")
        
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]
        
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)  # [num_edges, hidden_dim]
        
        
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr, batch)
        
        logits = self.classification_head(x)  # [num_nodes, 558]

        cell_type_logits = None
        if self.cell_type_head is not None:
            cell_type_logits = self.cell_type_head(x)  # [num_nodes, 7]

        batch_size = batch.max().item() + 1
        
        max_nodes = 0
        for b in range(batch_size):
            num_nodes_in_graph = (batch == b).sum().item()
            max_nodes = max(max_nodes, num_nodes_in_graph)
        
        batched_logits = torch.zeros(
            batch_size, max_nodes, self.config.out_dim,
            device=logits.device, dtype=logits.dtype
        )
        
        # Fill in batched tensors
        for b in range(batch_size):
            mask = (batch == b)
            num_nodes_in_graph = mask.sum().item()
            if num_nodes_in_graph > 0:
                batched_logits[b, :num_nodes_in_graph] = logits[mask]
        
        outputs = {
            "logits": batched_logits,
        }
        if cell_type_logits is not None:
            batched_ct = torch.zeros(batch_size, max_nodes, self.cell_type_head[-1].out_features,
                                     device=logits.device, dtype=logits.dtype)
            for b in range(batch_size):
                mask = (batch == b)
                n = mask.sum().item()
                if n > 0:
                    batched_ct[b, :n] = cell_type_logits[mask]
            outputs["cell_type_logits"] = batched_ct
        
        return outputs
