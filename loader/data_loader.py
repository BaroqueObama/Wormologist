import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Union, Optional, Iterator
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Data, Batch
import random

from loader.coordinate_encoder import CoordinateEncoder
from loader.augmentation import DataAugmentation
from loader.node_dropper import CurriculumNodeDropper
from loader.cell_types import CELL_TYPES, NUM_CELL_TYPES, CELL_TYPES_FINE


class CElegansShardedDataset(IterableDataset):
    """Sharded dataset for C. elegans that loads one HDF5 file at a time in CPU mem and distributes to GPUs for graph computation."""
    
    def __init__(self, 
                 data_path: Union[str, Path], # Path to data directory containing split subdirectories
                 split: str = "train", # Dataset split ("train", "val", "test")
                 coordinate_system: str = "cylindrical", # Coordinate system to use ("cartesian", "cylindrical")
                 normalize_coords: bool = False, # Whether to normalize coordinates
                 use_cell_type_features: bool = True, # Whether to include one-hot encoded cell type features
                 augmentation_config: Optional[object] = None, # AugmentationConfig object for data augmentation
                 curriculum_config: Optional[object] = None, # CurriculumConfig object for node dropping
                 shuffle_shards: bool = True, # Whether to shuffle order of shards
                 shuffle_within_shard: bool = True, # Whether to shuffle specimens within each shard
                 rank: Optional[int] = None, # Current process rank for distributed training
                 world_size: Optional[int] = None, # Total number of processes for distributed training
                 verbose: bool = False,
                 target: str = "canonical"): # Whether to print loading progress

        self.data_path = Path(data_path)
        self.split = split
        self.use_cell_type_features = use_cell_type_features and target == "canonical"
        self.coordinate_encoder = CoordinateEncoder(coordinate_system=coordinate_system, normalize=normalize_coords)
        
        self.augmentation = None
        if augmentation_config is not None:
            self.augmentation = DataAugmentation(augmentation_config)
        
        self.node_dropper = None
        if curriculum_config is not None:
            self.node_dropper = CurriculumNodeDropper(curriculum_config)
        
        self.current_global_batch = None  # Will be set by CurriculumCallback
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1
        self.verbose = verbose
        self.specimen_counter = 0
        self.split = split
        self.target = target
        
        split_dir = self.data_path / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.shard_files = sorted(list(split_dir.glob("*.h5")) + list(split_dir.glob("*.hdf5")))
        
        if not self.shard_files:
            raise ValueError(f"No HDF5 files found in {split_dir}")
        
        if self.verbose:
            print(f"Rank {self.rank}/{self.world_size}: Processing {len(self.shard_files)} shards")
    
    
    def _load_specimen(self, shard_data: h5py.File, specimen_key: str) -> Data:
        """Load a single specimen from the current shard."""

        raw_data = shard_data['specimens'][specimen_key][:] # [num_visible_nuclei, 4] => [canonical_id, x, y, z]
        
        canonical_ids = torch.from_numpy(raw_data[:, 0]).long()
        coords_3d = torch.from_numpy(raw_data[:, 1:4]).float()
        
        if self.augmentation is not None and self.augmentation.should_augment(self.split):
            coords_3d = self.augmentation.augment(coords_3d, self.specimen_counter, self.split)
        
        # Apply curriculum node dropping if configured
        visibility_rate = 1.0  # Default to full visibility
        
        if self.node_dropper is not None:
            global_batch = (self.current_global_batch if self.split == "train" and self.current_global_batch is not None else 0)

            visibility_rate = self.node_dropper.get_visibility_rate(global_batch, self.split)
            if visibility_rate < 1.0 or self.node_dropper.strategy == "sliced":
                coords_3d, canonical_ids, _ = self.node_dropper.drop_nodes(coords_3d, canonical_ids, visibility_rate, global_batch, self.split,)

        self.specimen_counter += 1 # for reproducible augmentation
        transformed_coords = self.coordinate_encoder.transform(coords_3d)
        
        if self.use_cell_type_features:
            cell_types_indices = CELL_TYPES[canonical_ids.numpy()]
            cell_type_features = F.one_hot(torch.from_numpy(cell_types_indices).long(), num_classes=NUM_CELL_TYPES).float()
            
            node_features = torch.cat([transformed_coords, cell_type_features], dim=-1) # TODO: Ensure this is taken in properly with raw coords stuff
        else:
            node_features = transformed_coords

        # Choose labels based on task
        cell_type_labels = torch.from_numpy(CELL_TYPES_FINE[canonical_ids]).long()
        labels = cell_type_labels if self.target == "cell_type" else canonical_ids

        # TODO: Check if computing graphs on GPU is actually better
        data = Data(x=node_features, y=labels, num_nodes=node_features.shape[0], raw_coords=coords_3d, visibility_rate=visibility_rate)
        return data
    
    
    def _process_shard(self, shard_path: Path) -> Iterator[Data]:
        if self.verbose:
            print(f"Rank {self.rank}: Loading shard {shard_path.name}")
        
        with h5py.File(shard_path, 'r') as shard_data:
            if 'specimens' not in shard_data:
                print(f"Warning: No 'specimens' group in {shard_path}")
                return
            
            specimen_keys = list(shard_data['specimens'].keys())
            
            if self.shuffle_within_shard and self.split == "train":
                random.shuffle(specimen_keys)

            total_specimens = len(specimen_keys)
            chunk_size = (total_specimens + self.world_size - 1) // self.world_size  # Ceiling division
            start_idx = self.rank * chunk_size
            end_idx = min(start_idx + chunk_size, total_specimens)
            
            my_specimen_keys = specimen_keys[start_idx : end_idx]
            
            if self.verbose and my_specimen_keys:
                print(f"Rank {self.rank}: Processing specimens {start_idx} to {end_idx-1} ({len(my_specimen_keys)} total)")
            
            for specimen_key in my_specimen_keys:
                try:
                    yield self._load_specimen(shard_data, specimen_key)
                except Exception as e:
                    print(f"Warning: Failed to load specimen {specimen_key} from {shard_path}: {e}")
                    continue
        
        if self.verbose:
            print(f"Rank {self.rank}: Finished shard {shard_path.name}")
    
    
    def __iter__(self) -> Iterator[Data]:
        """Iterate through all shards."""

        shard_order = list(self.shard_files)
        if self.shuffle_shards and self.split == "train":
            random.shuffle(shard_order)
        
        for shard_path in shard_order:
            for specimen in self._process_shard(shard_path):
                yield specimen
    

    def __len__(self): # TODO: Make smarter by reading the json file that comes with the dataset.
        avg_specimens_per_shard = 16384
        return len(self.shard_files) * avg_specimens_per_shard


def collate_batch(batch_data: List[Data]) -> Dict:
    """Collate function for batching PyTorch Geometric Data objects."""

    batch = Batch.from_data_list(batch_data)
    
    batch_size = len(batch_data)
    max_nodes = max(data.x.shape[0] for data in batch_data)
    visible_masks = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    
    for i, data in enumerate(batch_data):
        num_nodes = data.x.shape[0]
        visible_masks[i, :num_nodes] = True
    
    return {
        'batch': batch,  # PyG batch with coords and labels
        'visible_mask': visible_masks  # [batch_size, max_nodes]
    }


def create_data_loader(data_path: Union[str, Path], # Path to data directory
                       split: str, # Dataset split ("train", "val", "test")
                       batch_size: int, # Batch size (micro-batch size for distributed training)
                       coordinate_system: str = "cylindrical", # Coordinate system to use ("cartesian", "cylindrical")
                       normalize_coords: bool = False, # Whether to normalize coordinates
                       use_cell_type_features: bool = True, # Whether to include cell type features
                       augmentation_config: Optional[object] = None, # AugmentationConfig object for data augmentation
                       num_workers: int = 0,  # Number of workers (0 for IterableDataset)
                       distributed: bool = False, # Whether using distributed training
                       rank: Optional[int] = None, # Current process rank (auto-detected if distributed=True) # TODO: weird
                       world_size: Optional[int] = None, # Total processes (auto-detected if distributed=True) # TODO: weird
                       shuffle_shards: bool = None, # Whether to shuffle shard order (default: True for train)
                       shuffle_within_shard: bool = None, # Whether to shuffle within shards (default: True for train)
                       target: str = "canonical", # Choose the task, canonical ID prediction or cell type prediction
                       **dataset_kwargs,
                       ) -> DataLoader: # Additional dataset arguments
    
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        if rank is None:
            rank = torch.distributed.get_rank()
        if world_size is None:
            world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    if shuffle_shards is None:
        shuffle_shards = (split == "train")
    if shuffle_within_shard is None:
        shuffle_within_shard = (split == "train")
    
    dataset = CElegansShardedDataset(data_path=data_path, split=split, coordinate_system=coordinate_system, normalize_coords=normalize_coords, use_cell_type_features=use_cell_type_features, augmentation_config=augmentation_config, shuffle_shards=shuffle_shards, shuffle_within_shard=shuffle_within_shard, rank=rank, world_size=world_size, target=target, **dataset_kwargs)
    
    # For IterableDataset, we don't use a sampler and num_workers should be 0
    # Each dataset instance handles its own sharding
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,  # Should be 0
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=(split == "train")
    )
