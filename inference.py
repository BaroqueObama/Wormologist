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
import pickle
import json

from dataclasses import asdict
import json
from config.config import AugmentationConfig
import math


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

    # This was added to make it work with the GPU
    if cell_types is not None:
        cell_types = cell_types.to(device)
    
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

# A simple function that does not work with the DataLoader
def test_inference(test_file_path, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model_from_checkpoint(checkpoint_path, device=device)

    test_path = Path(test_file_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Could not find test data at {test_path}")

    with test_path.open("rb") as handle:
        worm_samples = pickle.load(handle)

    all_results = []
    total_nodes = 0
    total_top1_correct = 0
    total_top3_correct = 0
    total_top5_correct = 0

    for worm_idx, sample in enumerate(worm_samples):
        if len(sample) != 2:
            raise ValueError(f"Unexpected sample structure at index {worm_idx}: {sample}")

        canonical_ids, coords = sample
        canonical_ids = np.asarray(canonical_ids, dtype=np.int64)
        coords = np.asarray(coords, dtype=np.float32)

        if canonical_ids.ndim != 1:
            raise ValueError(f"Canonical id array must be 1-D, got shape {canonical_ids.shape}")
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Coordinate array must be [N, 3], got shape {coords.shape}")
        if canonical_ids.size == 0:
            continue
        if canonical_ids.min() < 0 or canonical_ids.max() >= len(CELL_TYPES):
            raise ValueError(f"Canonical IDs out of range for cell type lookup at worm {worm_idx}")

        cell_types = CELL_TYPES[canonical_ids]
        result = predict_single_pointcloud(
            model,
            coords,
            device=device,
            cell_types=cell_types,
        )

        num_nodes = canonical_ids.shape[0]
        gt_ids = torch.from_numpy(canonical_ids).long()

        pred_tensor = result["predictions"].view(-1).long()[:num_nodes]
        logits_tensor = result["logits"]
        if logits_tensor.dim() == 3 and logits_tensor.shape[0] == 1:
            logits_tensor = logits_tensor[0]
        logits_tensor = logits_tensor[:num_nodes]

        worm_top1_correct = (pred_tensor == gt_ids).sum().item()

        k5 = min(5, logits_tensor.shape[-1])
        k3 = min(3, k5)
        topk = torch.topk(logits_tensor, k=k5, dim=-1).indices
        top3 = topk[:, :k3]

        worm_top3_correct = (top3 == gt_ids.unsqueeze(1)).any(dim=1).sum().item()
        worm_top5_correct = (topk == gt_ids.unsqueeze(1)).any(dim=1).sum().item()

        worm_metrics = {
            "num_nodes": num_nodes,
            "top1_accuracy": worm_top1_correct / num_nodes,
            "top3_accuracy": worm_top3_correct / num_nodes,
            "top5_accuracy": worm_top5_correct / num_nodes,
        }

        print(
            f"Worm {worm_idx}: top1 {worm_metrics['top1_accuracy']:.3%}, "
            f"top3 {worm_metrics['top3_accuracy']:.3%}, "
            f"top5 {worm_metrics['top5_accuracy']:.3%}"
        )

        all_results.append(
            {
                "canonical_ids": canonical_ids,
                "predictions": pred_tensor.clone(),
                "logits": logits_tensor.clone(),
                "metrics": worm_metrics,
            }
        )

        total_nodes += num_nodes
        total_top1_correct += worm_top1_correct
        total_top3_correct += worm_top3_correct
        total_top5_correct += worm_top5_correct

    if total_nodes == 0:
        raise ValueError(f"No valid samples found in {test_path}")

    metrics = {
        "top1_accuracy": total_top1_correct / total_nodes,
        "top3_accuracy": total_top3_correct / total_nodes,
        "top5_accuracy": total_top5_correct / total_nodes,
        "total_nodes": total_nodes,
        "num_worms": len(all_results),
    }

    print(
        f"Overall: top1 {metrics['top1_accuracy']:.3%}, "
        f"top3 {metrics['top3_accuracy']:.3%}, "
        f"top5 {metrics['top5_accuracy']:.3%} "
        f"over {metrics['total_nodes']} nuclei across {metrics['num_worms']} worms"
    )

    return {"samples": all_results, "metrics": metrics}

# # Helper to find all subgraph datasets
# def _discover_subgraph_datasets(subgraph_root: Union[str, Path]) -> List[Path]:
#     subgraph_root = Path(subgraph_root)
#     dataset_dirs: List[Path] = []
#     for candidate in sorted(subgraph_root.glob("subgraph_*")):
#         test_dir = candidate / "test"
#         if test_dir.is_dir():
#             dataset_dirs.append(candidate)
#     if not dataset_dirs:
#         raise ValueError(f"No subgraph_* directories with a test/ split found under {subgraph_root}")
#     return dataset_dirs


def _discover_subgraph_datasets(subgraph_root: Union[str, Path]) -> List[Path]:
    """
    Find every dataset folder directly under `subgraph_root` that exposes a `test/`
    split containing at least one HDF5 shard. Works for both
    `sliced_subgraphs_shift_*` and `dataset_*` layouts:

        subgraph_root/
            <any-name>/
                test/
                    *.h5
    """
    subgraph_root = Path(subgraph_root)
    dataset_dirs: List[Path] = []

    for candidate in sorted(subgraph_root.iterdir()):
        if not candidate.is_dir():
            continue
        test_dir = candidate / "test"
        if test_dir.is_dir() and any(test_dir.glob("*.h5")):
            dataset_dirs.append(candidate)

    if not dataset_dirs:
        raise ValueError(
            f"No dataset folders with a test/ split found under {subgraph_root}"
        )
    return dataset_dirs

def _discover_shifted_subgraph_datasets(subgraph_root: Union[str, Path]) -> List[Path]:
    """
    Find all sliced-subgraph datasets under `subgraph_root` following the pattern:

        subgraph_root/
            sliced_subgraphs_shift_<offset>/
                test/
                    *.h5

    Returns the list of directories (each one passed to `create_data_loader`).
    """
    subgraph_root = Path(subgraph_root)
    dataset_dirs: List[Path] = []

    for candidate in sorted(subgraph_root.glob("sliced_subgraphs_shift_*")):
        test_dir = candidate / "test"
        if test_dir.is_dir() and any(test_dir.glob("*.h5")):
            dataset_dirs.append(candidate)

    if not dataset_dirs:
        raise ValueError(
            f"No sliced_subgraphs_shift_* directories with a test/ split found under {subgraph_root}"
        )
    return dataset_dirs


# The "new" testing function that works with the DataLoader
# It can probably be improved in one for the following ways:
# - changing the config to be specific for the testing
# - increase the batch size depending of the subgraph size
# - fix the progress bar so that it has the right length
def run_subgraph_suite(
    subgraph_root: Union[str, Path],
    checkpoint_path: Union[str, Path],
    results_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)

    # # Check if this makes sense
    # #################################################
    # model.loss_fn.matcher.temperature = config.sinkhorn.final_temperature
    # model.matcher.temperature = config.sinkhorn.final_temperature
    # print(f"Eval Sinkhorn temperature: {model.matcher.temperature}")
    # #################################################
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        inference_mode=True,
    )

    results: Dict[str, Dict[str, float]] = {}

    results_file: Optional[Path] = None
    if results_path is not None:
        results_file = Path(results_path)
        if results_file.is_dir():
            results_file = results_file / "subgraph_metrics.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

    # This should be instantiated elsewhere/more modularly (maybe)
    
    # aug_cfg = AugmentationConfig(
    #     apply_to_splits=["test"],
    #     z_shift_range=[0.02, 0.02], # same as [0.02, 0.020000001]
    #     # z_shift_range=[1.0, 1.0], # was just a sanity check that the transformation is 
    #     # z_shift_range=[0.0, 0.0], # another sanity check, but there is slight difference due to: normalize_z_axis. Does this imply anything about my slicing? Does this z normalization to zero make sense with subgraphs?
    #     uniform_scale_range=[1.0, 1.0],
    #     z_rotation_range=[0.0, 0.0],
    #     post_rotation_xy_scale_range=[1.0, 1.0],
    #     xy_scale_orientation_range=[0.0, 0.0],
    # )
    # print(json.dumps(asdict(aug_cfg), indent=2))

    for dataset_dir in _discover_subgraph_datasets(subgraph_root):
        loader = create_data_loader(
            data_path=dataset_dir,
            split="test",
            batch_size=config.training.micro_batch_size,
            coordinate_system=config.data.coordinate_system,
            normalize_coords=config.data.normalize_coords,
            use_cell_type_features=config.data.use_cell_type_features,
            augmentation_config=None,  # aug_cfg,  # No augmentation during testing        
            num_workers=0,
            distributed=False,
            shuffle_shards=False,
            shuffle_within_shard=False,
        )

        metrics = trainer.test(model, dataloaders=loader, ckpt_path=None, verbose=False)
        split_metrics = metrics[0] if metrics else {}
        results[dataset_dir.name] = split_metrics
        print(f"{dataset_dir.name}: {split_metrics}")

        if results_file is not None:
            with results_file.open("w") as handle:
                json.dump(results, handle, indent=2)

    if results_file is not None and results:
        summary: Dict[str, float] = {}
        metric_keys = set().union(*(metrics.keys() for metrics in results.values()))
        for key in metric_keys:
            values = [
                float(split_metrics[key])
                for split_metrics in results.values()
                if isinstance(split_metrics.get(key), (int, float))
            ]
            if not values:
                continue
            mean_val = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
                std_val = variance ** 0.5
            else:
                std_val = 0.0
            summary[f"{key}/mean"] = mean_val
            summary[f"{key}/std"] = std_val

        # New aggregation goes here:
        acc_totals: Dict[str, Dict[str, float]] = {}
        for split_metrics in results.values():
            for name, value in split_metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                for suffix, bucket in [("_sum", "sum"), ("_sum_sq", "sum_sq"), ("_count", "count")]:
                    if name.endswith(suffix):
                        base = name[: -len(suffix)]
                        acc_totals.setdefault(base, {"sum": 0.0, "sum_sq": 0.0, "count": 0.0})
                        acc_totals[base][bucket] += float(value)
                        break

        for base, totals in acc_totals.items():
            count = totals["count"]
            total_sum = totals["sum"]
            total_sq = totals["sum_sq"]
            if count <= 0:
                continue
            mean_val = total_sum / count
            if count > 1:
                variance = (total_sq - (total_sum ** 2) / count) / (count - 1)
                std_val = math.sqrt(max(variance, 0.0))
            else:
                std_val = 0.0
            summary[f"{base}/global_mean"] = mean_val
            summary[f"{base}/global_std"] = std_val

        summary_file = results_file.with_name("metrics_summary.json")
        with summary_file.open("w") as handle:
            json.dump(summary, handle, indent=2)

    return results


def _compute_unbiased_std(total_sum: float, total_sq: float, count: float) -> float:
    if count <= 1:
        return 0.0
    variance = (total_sq - (total_sum ** 2) / count) / (count - 1)
    return math.sqrt(max(variance, 0.0))

def run_single_subgraph(
    subgraph_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    results_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Evaluate a single subgraph dataset (directory or individual shard) with the
    same machinery used by run_subgraph_suite.
    """
    dataset_path = Path(subgraph_path).expanduser().resolve()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model_from_checkpoint(checkpoint_path, device=device)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        inference_mode=True,
    )

    # Determine whether we were given a directory (with split subfolders) or a single shard.
    if dataset_path.suffix.lower() in {".h5", ".hdf5"}:
        split_name = dataset_path.parent.name              # e.g. "test"
        data_root = dataset_path.parent.parent             # directory that holds the split folders
        single_shard = dataset_path
    else:
        data_root = dataset_path
        split_name = "test"
        single_shard = None

    loader = create_data_loader(
        data_path=data_root,
        split=split_name,
        batch_size=config.training.micro_batch_size,
        coordinate_system=config.data.coordinate_system,
        normalize_coords=config.data.normalize_coords,
        use_cell_type_features=config.data.use_cell_type_features,
        augmentation_config=None,
        num_workers=0,
        distributed=False,
        shuffle_shards=False,
        shuffle_within_shard=False,
    )

    if single_shard is not None:
        loader.dataset.shard_files = [single_shard]

    metrics_list = trainer.test(model, dataloaders=loader, ckpt_path=None, verbose=False)
    metrics = metrics_list[0] if metrics_list else {}
    print(f"{dataset_path.name}: {metrics}")
    
    if results_path is not None:
        results_file = Path(results_path)
        if results_file.is_dir():
            results_file = results_file / f"{dataset_path.stem}_metrics.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with results_file.open("w") as handle:
            json.dump(metrics, handle, indent=2)

        # write a metrics summary like run_subgraph_suite
        summary = {
            "test/exact_accuracy/global_mean": metrics["test/exact_accuracy_sum"] / metrics["test/exact_accuracy_count"],
            "test/exact_accuracy/global_std": _compute_unbiased_std(
                metrics["test/exact_accuracy_sum"],
                metrics["test/exact_accuracy_sum_sq"],
                metrics["test/exact_accuracy_count"],
            ),
            "test/top3_accuracy/global_mean": metrics["test/top3_accuracy_sum"] / metrics["test/top3_accuracy_count"],
            "test/top3_accuracy/global_std": _compute_unbiased_std(
                metrics["test/top3_accuracy_sum"],
                metrics["test/top3_accuracy_sum_sq"],
                metrics["test/top3_accuracy_count"],
            ),
            "test/top5_accuracy/global_mean": metrics["test/top5_accuracy_sum"] / metrics["test/top5_accuracy_count"],
            "test/top5_accuracy/global_std": _compute_unbiased_std(
                metrics["test/top5_accuracy_sum"],
                metrics["test/top5_accuracy_sum_sq"],
                metrics["test/top5_accuracy_count"],
            ),
        }

        summary_file = results_file.with_name("metrics_summary.json")
        with summary_file.open("w") as handle:
            json.dump(summary, handle, indent=2)

    return metrics

if __name__ == "__main__":
    # test_file_path = "/fs/pool/pool-mlsb/bulat/Wormologist/graph_matching/sliced_subgraphs_for_inference.pkl"
    # # checkpoint_path = "/fs/pool/pool-mlsb/bulat/Wormologist/Baseline_Multiscale/Baseline_Multiscale/last.ckpt"
    # checkpoint_path = "/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types/Three_Cell_Types/last.ckpt"

    # test_inference(test_file_path, checkpoint_path)
    # subgraph_root = "/fs/pool/pool-mlsb/bulat/Wormologist/new_subgraph_testing_data_sliced_shift_projected_20_slices"
    # checkpoint_path = "/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types/Three_Cell_Types/last.ckpt"
    # results_path = "/fs/pool/pool-mlsb/bulat/Wormologist/redefined_testing_results/3_cell_types_shift_20_slices_projected/metrics.json"

    # run_subgraph_suite(subgraph_root, checkpoint_path, results_path)

    # run_single_subgraph(
    # "/fs/pool/pool-mlsb/bulat/Wormologist/new_subgraph_testing_data_sliced/sliced_subgraphs/test/test_0000.h5",
    # checkpoint_path="/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types/Three_Cell_Types/last.ckpt",)
    
    # checkpoint_path = Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_one_to_one_sliced_data_2/Three_Cell_Types_one_to_one_sliced_data_2/last.ckpt")
    # parent_root = Path("/fs/pool/pool-mlsb/bulat/Wormologist/real_test_set")

    # for subgraph_dir in sorted(parent_root.iterdir()):
    #     if not subgraph_dir.is_dir():
    #         continue  # skip stray files

    #     subgraph_root = subgraph_dir
    #     results_path = subgraph_dir / "results_one_to_one_sliced_data_2" / "metrics.json"

    #     results_path.parent.mkdir(parents=True, exist_ok=True)

    #     run_subgraph_suite(
    #         subgraph_root=subgraph_root,
    #         checkpoint_path=checkpoint_path,
    #         results_path=results_path,
    #     )

#     run_single_subgraph(
#     subgraph_path=Path("/fs/pool/pool-mlsb/bulat/Wormologist/new_subgraph_testing_data/subgraph_558"),
#     checkpoint_path=Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_one_to_one_sliced_data/Three_Cell_Types_one_to_one_sliced_data/last.ckpt"),
#     results_path=Path("/fs/pool/pool-mlsb/bulat/Wormologist/new_subgraph_testing_data/subgraph_558/metrics_one_to_one_sliced_data.json"),
# )

    run_subgraph_suite(
    subgraph_root=Path("/fs/pool/pool-mlsb/bulat/Wormologist/random_comparison_to_real_test_set_40"),
    checkpoint_path=Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_three_cell_types_one_to_one_sliced_data/Three_Cell_Types_one_to_one_sliced_data/last.ckpt"),
    results_path=Path("/fs/pool/pool-mlsb/bulat/Wormologist/random_comparison_to_real_test_set_40/one_to_one_sliced_data/metrics_with_hungarian3.json"),
)