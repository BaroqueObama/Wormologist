#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from torch_geometric.data import Data, Batch

from config.config import Config
from loader.data_loader import create_data_loader, collate_batch
from loader.coordinate_encoder import CoordinateEncoder
from loader.graph_encoder import GraphEncoder
from loader.cell_types import CELL_TYPES, NUM_CELL_TYPES
from model.lightning_model import LightningModel


def handle_state_dict_gpu_mismatch(state_dict: dict, remove_module_prefix: bool = True) -> dict:
    """Handle state dict from multi-GPU training for single GPU or different GPU inference."""
    
    if not remove_module_prefix:
        return state_dict
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix added by DataParallel/DistributedDataParallel # TODO: better handling
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    return new_state_dict


def load_model_from_checkpoint(checkpoint_path: str, config: Config = None, device: str = 'cuda', map_location: str = None) -> Tuple[LightningModel, Config]:
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        config: Optional config to override checkpoint config
        device: Target device ('cuda', 'cpu', 'cuda:0', etc)
        map_location: Device mapping for loading (defaults to device)
    
    Returns:
        Tuple of (Loaded LightningModel, Config used)
    """
    if map_location is None:
        map_location = device if device != 'cuda' or not torch.cuda.is_available() else None
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    if config is None:
        if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
            config = checkpoint['hyper_parameters']['config']
        else:
            raise ValueError("No config found in checkpoint and none provided")
    

    model = LightningModel(config)
    

    state_dict = checkpoint['state_dict']
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        if 'Missing key(s)' in str(e) or 'Unexpected key(s)' in str(e):
            cleaned_state_dict = handle_state_dict_gpu_mismatch(state_dict) # Try removing module prefix # TODO: handle better
            model.load_state_dict(cleaned_state_dict, strict=False)
            print("Loaded checkpoint with modified state dict (removed module prefix)")
    
    model = model.to(device)
    model.eval()
    
    return model, config


def prepare_raw_pointcloud(coords_3d: torch.Tensor, config: Config, device: str = 'cuda', cell_types: Optional[torch.Tensor] = None) -> Dict:
    if config.data.use_cell_type_features and cell_types is None:
        raise RuntimeError("Model was trained with cell type features but none provided.")
    

    if coords_3d.dim() == 2:
        coords_3d = coords_3d.unsqueeze(0)
        if cell_types is not None and cell_types.dim() == 1:
            cell_types = cell_types.unsqueeze(0)
        single_sample = True
    else:
        single_sample = False
    
    batch_size = coords_3d.shape[0]
    
    coords_3d = coords_3d.to(device)
    
    coordinate_encoder = CoordinateEncoder(coordinate_system=config.data.coordinate_system, normalize=config.data.normalize_coords)
    
    # Create PyG Data objects for each sample
    data_list = []
    
    for b in range(batch_size):
        sample_coords = coords_3d[b]
        # Remove padding by checking for invalid z-coordinates
        # C. elegans data has valid (0,0,0) points but no points with z < -0.5
        # This filters out padding
        valid_mask = sample_coords[:, 2] >= -0.5
        sample_coords = sample_coords[valid_mask]
        
        if len(sample_coords) == 0:
            continue
        
        transformed = coordinate_encoder.transform(sample_coords)
        
        if config.data.use_cell_type_features:
            sample_cell_types = cell_types[b][valid_mask]
            if sample_cell_types.dim() == 1:
                sample_cell_types = F.one_hot(sample_cell_types.long(), num_classes=NUM_CELL_TYPES).float()
            else:
                if sample_cell_types.shape[1] != NUM_CELL_TYPES:
                    raise ValueError(f"One-hot cell types must have {NUM_CELL_TYPES} dimensions, got {sample_cell_types.shape[1]}")
                sample_cell_types = sample_cell_types.float()
            
            transformed = torch.cat([transformed, sample_cell_types], dim=-1)
        
        data = Data(
            x=transformed,
            raw_coords=sample_coords,
            num_nodes=len(transformed)
        )
        data_list.append(data)
    
    if not data_list:
        raise ValueError("No valid coordinates found in input point cloud.")
    
    batch_dict = collate_batch(data_list)
    batch_dict['single_sample'] = single_sample
    
    return batch_dict


def predict_single_pointcloud(model: LightningModel, coords_3d: Union[np.ndarray, torch.Tensor], device: str = 'cuda', cell_types: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict:
    """
    Run inference on a single 3D point cloud.
    
    Args:
        model: Trained Lightning model
        coords_3d: 3D coordinates as numpy array or torch tensor [N, 3] or [batch_size, N, 3]
        device: Device to run inference on
        cell_types: Optional cell type labels [N] with values 0-6
    
    Returns:
        Dictionary with predictions, confidence scores, and logits
    """

    if isinstance(coords_3d, np.ndarray):
        coords_3d = torch.from_numpy(coords_3d).float()
    
    if cell_types is not None and isinstance(cell_types, np.ndarray):
        cell_types = torch.from_numpy(cell_types)
    

    batch = prepare_raw_pointcloud(coords_3d, model.config, device, cell_types)
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(batch)
        
        visible_mask = batch['visible_mask']
        log_probs = F.log_softmax(outputs['logits'], dim=-1)
        assignments = model.matcher(log_probs, visible_mask)
        
        predictions = assignments['hard_assignments']
        logits = outputs['logits']
        
        if batch.get('single_sample', False):
            predictions = predictions[0] if predictions.dim() > 1 else predictions
            logits = logits[0] if logits.dim() > 1 else logits
    
    return {
        'predictions': predictions.cpu(),
        'logits': logits.cpu()
    }


# Example usage
def example_raw_pointcloud_inference():

    # Generate example point cloud
    num_points = 100
    coords_3d = np.random.randn(num_points, 3) * 10
    
    # Load model
    checkpoint_path = "checkpoints/model.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)
    
    # Run inference
    results = predict_single_pointcloud(model, coords_3d, device)
    
    print(f"Predictions: {results['predictions']}")
    
    return results