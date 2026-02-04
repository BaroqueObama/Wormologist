#!/usr/bin/env python
from dataclasses import asdict
from pathlib import Path
import torch
import pytorch_lightning as L
from typing import Dict, Optional, Union, List, Tuple, Any

from config.config import AugmentationConfig, Config, TaskConfig
from loader.data_loader import create_data_loader
from model.lightning_model import LightningModel
import json
import math

from loader.cell_types import CELL_TYPES, NUM_CELL_TYPES, CELL_TYPES_FINE, NUM_CELL_TYPES_FINE
import numpy as np


# at module scope
METRICS_TO_SAVE = {
    # fill in the exact metric names you want
    "test/accuracy",
    "test/accuracy_hungarian",
    "test/accuracy_logits",
    "test/top5_accuracy",
    "test/top3_accuracy"
}

def _filter_metrics(metrics: Dict[str, float], keep: set[str]) -> Dict[str, float]:
    if not keep:
        return metrics  # empty set means keep everything
    return {k: v for k, v in metrics.items() if k in keep}

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
        
    # TODO: check if this is fine. I have to make the old models without TaskConfig compatible.
    if isinstance(config, dict):
        config = Config.from_dict(config)  # normalize old dict configs
    if not hasattr(config, "task") or config.task is None:
        config.task = TaskConfig()        # backward-compat default

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

def compute_cell_type_error_stats_for_loader(
    model: LightningModel,
    loader,
    device: str = "cuda",
    cell_types: np.ndarray = CELL_TYPES,
    num_cell_types: int = NUM_CELL_TYPES,
) -> Dict[str, Any]:
    """
    Run model on a dataloader and compute cell-type-specific error statistics.

    Uses canonical IDs from pyg_batch.y, mapped to cell types via CELL_TYPES.
    """

    model.eval()

    total_per_type = np.zeros(num_cell_types, dtype=np.int64)
    correct_per_type = np.zeros(num_cell_types, dtype=np.int64)
    confusion = np.zeros((num_cell_types, num_cell_types), dtype=np.int64)

    same_type_diff_id = 0
    diff_type_errors = 0

    with torch.no_grad():
        for batch in loader:
            # Move tensors to device for forward pass
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v

            pyg_batch = batch_device["batch"]
            labels = pyg_batch.y.long()                  # canonical IDs per node
            batch_assignment = pyg_batch.batch.long()
            visible_mask = batch_device["visible_mask"]  # [B, N] on device

            # Prepare targets as in LightningModel
            targets = model.prepare_targets(labels, batch_assignment, visible_mask)

            # Run the model and use the same Sinkhorn/SuperGlue matcher used in test_step
            outputs = model(batch_device)  # {"logits": ...}
            device_logits = outputs["logits"].device
            targets["labels"] = targets["labels"].to(device_logits)
            targets["visible_mask"] = targets["visible_mask"].to(device_logits)

            loss_dict = model.loss_fn(outputs, targets)  # includes "soft_assignments"
            preds = loss_dict["soft_assignments"].argmax(dim=-1)  # [B, N]

            # Move to CPU for stats
            labels_cpu = targets["labels"].cpu()
            visible_mask_cpu = targets["visible_mask"].cpu()
            preds_cpu = preds.cpu()

            B, N = preds_cpu.shape
            for b in range(B):
                vis = visible_mask_cpu[b]                # [N] bool on CPU
                vis_idx = torch.where(vis)[0]            # CPU indices
                if vis_idx.numel() == 0:
                    continue

                true_ids_b = labels_cpu[b, vis_idx].numpy()
                pred_ids_b = preds_cpu[b, vis_idx].numpy()

                true_ct = cell_types[true_ids_b]
                pred_ct = cell_types[pred_ids_b]

                # Totals and correctness per true cell type
                for ct in range(num_cell_types):
                    mask_ct = (true_ct == ct)
                    if not mask_ct.any():
                        continue
                    total_ct = int(mask_ct.sum())
                    total_per_type[ct] += total_ct
                    correct_per_type[ct] += int((mask_ct & (true_ids_b == pred_ids_b)).sum())

                # Mispredictions
                mis_mask = (true_ids_b != pred_ids_b)
                if not mis_mask.any():
                    continue

                true_ct_err = true_ct[mis_mask]
                pred_ct_err = pred_ct[mis_mask]
                for t, p in zip(true_ct_err, pred_ct_err):
                    confusion[int(t), int(p)] += 1

                same_mask = (true_ct_err == pred_ct_err)
                same_type_diff_id += int(same_mask.sum())
                diff_type_errors += int((~same_mask).sum())

    total_nodes = int(total_per_type.sum())
    total_mispreds = int(same_type_diff_id + diff_type_errors)

    per_type_summary: Dict[str, Any] = {}
    for ct in range(num_cell_types):
        total_ct = int(total_per_type[ct])
        correct_ct = int(correct_per_type[ct])
        mis_ct = total_ct - correct_ct
        acc_ct = (correct_ct / total_ct) if total_ct > 0 else 0.0
        frac_nodes = (total_ct / total_nodes) if total_nodes > 0 else 0.0
        frac_errors = (mis_ct / total_mispreds) if total_mispreds > 0 else 0.0

        per_type_summary[str(ct)] = {
            "total_nodes": total_ct,
            "correct": correct_ct,
            "mispredicted": mis_ct,
            "accuracy": acc_ct,
            "fraction_of_all_nodes": frac_nodes,
            "fraction_of_all_errors": frac_errors,
        }

    row_norm_conf = np.zeros_like(confusion, dtype=np.float64)
    for ct in range(num_cell_types):
        row_sum = confusion[ct].sum()
        if row_sum > 0:
            row_norm_conf[ct] = confusion[ct] / row_sum

    error_type_breakdown = {
        "total_mispredictions": total_mispreds,
        "same_cell_type_wrong_id": same_type_diff_id,
        "different_cell_type": diff_type_errors,
        "same_cell_type_wrong_id_fraction": (
            same_type_diff_id / total_mispreds if total_mispreds > 0 else 0.0
        ),
        "different_cell_type_fraction": (
            diff_type_errors / total_mispreds if total_mispreds > 0 else 0.0
        ),
    }

    return {
        "num_cell_types": num_cell_types,
        "per_cell_type": per_type_summary,
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_row_normalized": row_norm_conf.tolist(),
        "error_type_breakdown": error_type_breakdown,
        "total_nodes": total_nodes,
    }

def _init_celltype_totals(num_cell_types: int = NUM_CELL_TYPES) -> Dict[str, Any]:
    return {
        "total_per_type": np.zeros(num_cell_types, dtype=np.int64),
        "correct_per_type": np.zeros(num_cell_types, dtype=np.int64),
        "confusion": np.zeros((num_cell_types, num_cell_types), dtype=np.int64),
        "same_type_diff_id": 0,
        "diff_type_errors": 0,
        "total_nodes": 0,
    }

def _accumulate_celltype_totals(totals: Dict[str, Any], ct_stats: Dict[str, Any]) -> None:
    num_cell_types = len(ct_stats["per_cell_type"])
    per = ct_stats["per_cell_type"]
    totals["total_per_type"] += np.array([per[str(i)]["total_nodes"] for i in range(num_cell_types)], dtype=np.int64)
    totals["correct_per_type"] += np.array([per[str(i)]["correct"] for i in range(num_cell_types)], dtype=np.int64)
    totals["confusion"] += np.array(ct_stats["confusion_matrix_counts"], dtype=np.int64)
    totals["same_type_diff_id"] += ct_stats["error_type_breakdown"]["same_cell_type_wrong_id"]
    totals["diff_type_errors"] += ct_stats["error_type_breakdown"]["different_cell_type"]
    totals["total_nodes"] += ct_stats["total_nodes"]

def _celltype_summary_from_totals(totals: Dict[str, Any]) -> Dict[str, Any]:

    num_cell_types = totals["total_per_type"].shape[0]

    total_nodes = int(totals["total_per_type"].sum())
    total_mispreds = int(totals["same_type_diff_id"] + totals["diff_type_errors"])

    per_type_summary: Dict[str, Any] = {}
    for ct in range(num_cell_types):
        total_ct = int(totals["total_per_type"][ct])
        correct_ct = int(totals["correct_per_type"][ct])
        mis_ct = total_ct - correct_ct
        per_type_summary[str(ct)] = {
            "total_nodes": total_ct,
            "correct": correct_ct,
            "mispredicted": mis_ct,
            "accuracy": (correct_ct / total_ct) if total_ct > 0 else 0.0,
            "fraction_of_all_nodes": (total_ct / total_nodes) if total_nodes > 0 else 0.0,
            "fraction_of_all_errors": (mis_ct / total_mispreds) if total_mispreds > 0 else 0.0,
        }

    confusion = totals["confusion"]
    row_norm_conf = np.zeros_like(confusion, dtype=np.float64)
    for ct in range(num_cell_types):
        row_sum = confusion[ct].sum()
        if row_sum > 0:
            row_norm_conf[ct] = confusion[ct] / row_sum

    error_breakdown = {
        "total_mispredictions": total_mispreds,
        "same_cell_type_wrong_id": int(totals["same_type_diff_id"]),
        "different_cell_type": int(totals["diff_type_errors"]),
        "same_cell_type_wrong_id_fraction": (totals["same_type_diff_id"] / total_mispreds) if total_mispreds > 0 else 0.0,
        "different_cell_type_fraction": (totals["diff_type_errors"] / total_mispreds) if total_mispreds > 0 else 0.0,
    }

    return {
        "num_cell_types": num_cell_types,
        "per_cell_type": per_type_summary,
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_row_normalized": row_norm_conf.tolist(),
        "error_type_breakdown": error_breakdown,
        "total_nodes": total_nodes,
    }

def _has_h5_shards(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.h5"))

def _discover_subgraph_datasets(subgraph_root: Union[str, Path]) -> List[Path]:
    """
    Find dataset folders under `subgraph_root` that expose a `test/` split with at
    least one HDF5 shard. Supports both nested shift layouts and single-level
    datasets such as the random subgraphs.
    """

    subgraph_root = Path(subgraph_root)
    dataset_dirs: List[Path] = []

    if _has_h5_shards(subgraph_root / "test"):
        dataset_dirs.append(subgraph_root)

    for candidate in sorted(subgraph_root.iterdir()):
        if not candidate.is_dir():
            continue

        if candidate.name == "test" and _has_h5_shards(candidate):
            if subgraph_root not in dataset_dirs:
                dataset_dirs.append(subgraph_root)
            continue

        test_dir = candidate / "test"
        if _has_h5_shards(test_dir) and candidate not in dataset_dirs:
            dataset_dirs.append(candidate)    

    if not dataset_dirs:
        raise ValueError(
            f"No dataset folders with a test/ split found under {subgraph_root}"
        )
    return dataset_dirs

def _summarize_metrics(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    if not results:
        return summary

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
            std_val = math.sqrt(max(variance, 0.0))
        else:
            std_val = 0.0
        summary[f"{key}/mean"] = mean_val
        summary[f"{key}/std"] = std_val

    acc_totals: Dict[str, Dict[str, float]] = {}
    for split_metrics in results.values():
        for name, value in split_metrics.items():
            if not isinstance(value, (int, float)):
                continue
            for suffix, bucket in [("_sum", "sum"), ("_sum_sq", "sum_sq"), ("_count", "count")]:
                if name.endswith(suffix):
                    totals = acc_totals.setdefault(
                        name[: -len(suffix)],
                        {"sum": 0.0, "sum_sq": 0.0, "count": 0.0},
                    )
                    totals[bucket] += float(value)
                    break

    for base, totals in acc_totals.items():
        count = totals["count"]
        total_sum = totals["sum"]
        total_sq = totals["sum_sq"]
        if count <= 0:
            continue
        mean_val = total_sum / count
        std_val = _compute_unbiased_std(total_sum, total_sq, count)
        summary[f"{base}/global_mean"] = mean_val
        summary[f"{base}/global_std"] = std_val

    return summary

# The "new" testing function that works with the DataLoader
# It can probably be improved in one for the following ways:
# - changing the config to be specific for the testing
# - increase the batch size depending of the subgraph size
# - fix the progress bar so that it has the right length
def run_subgraph_suite(
    subgraph_root: Union[str, Path],
    checkpoint_path: Union[str, Path],
    results_path: Optional[Union[str, Path]] = None,
    compute_cell_type_metrics: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Optional[Dict[str, Any]]]:
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

    results_file: Optional[Path] = None
    if results_path is not None:
        results_file = Path(results_path)
        if results_file.is_dir():
            results_file = results_file / "subgraph_metrics.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

    # This should be instantiated elsewhere/more modularly (maybe)

    ###########################
    
    # aug_cfg = AugmentationConfig(
    #     apply_to_splits=["test"],
    #     z_shift_range=[0.02, 0.02], # same as [0.02, 0.020000001]
    #     # z_shift_range=[1.0, 1.0], # was just a sanity check that the transformation is 
    #     # z_shift_range=[0.0, 0.0], # another sanity check, but there is slight difference due to: normalize_z_axis. Does this imply anything about my slicing? Does this z normalization to zero make sense with subgraphs?
    #     uniform_scale_range=[1.0, 1.0],
    #     z_rotation_range=[0.0, 0.0],
    #     post_rotation_xy_scale_range=[1.0, 1.0],
    #     xy_scale_orientation_range=[0.0, 0.0],
    #     normalize_z_axis=False
    # )
    # print(json.dumps(asdict(aug_cfg), indent=2))

    ###########################

    results_core: Dict[str, Dict[str, float]] = {}
    results_full: Dict[str, Dict[str, float]] = {}

    ct_totals_fine = _init_celltype_totals(NUM_CELL_TYPES_FINE) if compute_cell_type_metrics else None

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
        results_full[dataset_dir.name] = split_metrics


        filtered_metrics = _filter_metrics(split_metrics, METRICS_TO_SAVE)
        results_core[dataset_dir.name] = filtered_metrics
        print(f"{dataset_dir.name}: {filtered_metrics}")

         # Cell-type error analysis for this dataset
        if compute_cell_type_metrics:
            ct_stats_fine = compute_cell_type_error_stats_for_loader(
                model, loader, device=device, cell_types=CELL_TYPES_FINE, num_cell_types=NUM_CELL_TYPES_FINE
            )
            if results_file is not None:
                ct_out_path = results_file.with_name(f"{dataset_dir.name}_celltype_errors.json")
                with ct_out_path.open("w") as handle:
                    json.dump(ct_stats_fine, handle, indent=2)

            _accumulate_celltype_totals(ct_totals_fine, ct_stats_fine)

        if results_file is not None:
            core_path = results_file.with_name(f"{results_file.stem}_core{results_file.suffix}")
            full_path = results_file.with_name(f"{results_file.stem}_full{results_file.suffix}")
            core_path.parent.mkdir(parents=True, exist_ok=True)
            with core_path.open("w") as h:
                json.dump(results_core, h, indent=2)
            with full_path.open("w") as h:
                json.dump(results_full, h, indent=2)

    if results_file is not None and results_core:
        summary_core = _summarize_metrics(results_core)
        summary_full = _summarize_metrics(results_full)
        core_summary = results_file.with_name("metrics_summary_core.json")
        full_summary = results_file.with_name("metrics_summary_full.json")
        with core_summary.open("w") as h:
            json.dump(summary_core, h, indent=2)
        with full_summary.open("w") as h:
            json.dump(summary_full, h, indent=2)

        if compute_cell_type_metrics and ct_totals_fine is not None:
            ct_summary = _celltype_summary_from_totals(ct_totals_fine)
            ct_summary_path = results_file.with_name("celltype_errors_summary.json")
            with ct_summary_path.open("w") as h:
                json.dump(ct_summary, h, indent=2)

    return results_core, results_full, ct_totals_fine


def _compute_unbiased_std(total_sum: float, total_sq: float, count: float) -> float:
    if count <= 1:
        return 0.0
    variance = (total_sq - (total_sum ** 2) / count) / (count - 1)
    return math.sqrt(max(variance, 0.0))

# if __name__ == "__main__":
    
    # checkpoint_path = Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types/Three_Cell_Types/last.ckpt")
    # parent_root = Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/real_test_set_presentation/40_slices")

    # results_dir_name = "presentation_results_many_to_one_random_data_data_aug_toggle"
    # all_results: Dict[str, Dict[str, float]] = {}

    # for subgraph_dir in sorted(parent_root.iterdir()):
    #     if not subgraph_dir.is_dir():
    #         continue  # skip stray files
    #     if not (subgraph_dir / "test").is_dir():
    #         continue

    #     subgraph_root = subgraph_dir
    #     results_path = subgraph_dir / results_dir_name / "metrics.json"

    #     results_path.parent.mkdir(parents=True, exist_ok=True)

    #     suite_results = run_subgraph_suite(
    #         subgraph_root=subgraph_root,
    #         checkpoint_path=checkpoint_path,
    #         results_path=results_path,
    #     )

    #     for dataset_name, metrics in suite_results.items():
    #         all_results[f"{subgraph_dir.name}/{dataset_name}"] = metrics

    # if all_results:
    #     overall_dir = parent_root / results_dir_name
    #     overall_dir.mkdir(parents=True, exist_ok=True)
    #     overall_summary_path = overall_dir / "overall_metrics_summary.json"
    #     overall_summary = _summarize_metrics(all_results)
    #     with overall_summary_path.open("w") as handle:
    #         json.dump(overall_summary, handle, indent=2)

# Experiments Suite

if __name__ == "__main__":

    compute_cell_type_metrics = False

    parent_roots = [
        # Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/real_test_set_presentation/20_slices"),
        Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/real_test_set_presentation/40_slices"),
        # Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/random_comparison_to_real_test_presentation/20_slices"),
        # Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/random_comparison_to_real_test_presentation/40_slices"),
    ]

    # experiments = [
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types/Three_Cell_Types/last.ckpt"),
    #         "results_dir_name": "many_to_one_random_data",
    #     },
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_one_to_one_fixed/Three_Cell_Types_One_to_One_Fixed/last.ckpt"),
    #         "results_dir_name": "one_to_one_random_data",
    #     },
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_many_to_one_sliced_data/Three_Cell_Types_many_to_one_sliced_data/last.ckpt"),
    #         "results_dir_name": "many_to_one_sliced_data",
    #     },
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_one_to_one_sliced_data/Three_Cell_Types_one_to_one_sliced_data/last.ckpt"),
    #         "results_dir_name": "one_to_one_sliced_data",
    #     },
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_many_to_one_uniform_finetune/Three_Cell_Types_many_to_one_uniform_finetune/last.ckpt"),
    #         "results_dir_name": "many_to_one_uniform_finetune",
    #     },
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_many_to_one_curriculum_finetune/Three_Cell_Types_many_to_one_curriculum_finetune/last.ckpt"),
    #         "results_dir_name": "many_to_one_curriculum_finetune",
    #     },
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_one_to_one_uniform_finetune/Three_Cell_Types_one_to_one_uniform_finetune/last.ckpt"),
    #         "results_dir_name": "one_to_one_uniform_finetune",
    #     },
    #     {
    #         "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/checkpoint_three_cell_types_one_to_one_curriculum_finetune/Three_Cell_Types_one_to_one_curriculum_finetune/last.ckpt"),
    #         "results_dir_name": "one_to_one_curriculum_finetune",
    #     },
    # ]

    experiments = [
        {
            "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_no_cell_types_one_to_one_sliced_data_superglue_loss_trainable_dustbins/no_cell_types_one_to_one_sliced_data_superglue_loss_trainable_dustbins/epoch=epoch=0-val_accuracy=val/accuracy=0.8231.ckpt"),
            "results_dir_name": "no_cell_types_superglue_loss_trainable_dustbins_best_val_ckpt",
        },
        # {
        #     "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_superglue_loss_superglue_finetuned_top5/superglue_loss_superglue_finetuned_top5/epoch=epoch=0-val_accuracy=val/accuracy=0.8856.ckpt"),
        #     "results_dir_name": "superglue_loss_superglue_finetuned_top5_best_val_ckpt",
        # },
        # {
        #     "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_one_to_one_sliced_data_superglue_finetuned_top5/one_to_one_sliced_data_superglue_finetuned_top5/epoch=epoch=0-val_accuracy=val/accuracy=0.8854.ckpt"),
        #     "results_dir_name": "one_to_one_sliced_data_superglue_finedtuned_top5_best_val_ckpt",
        # },
        # {
        #     "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_one_to_one_sliced_data_superglue_finetuned_top10/one_to_one_sliced_data_superglue_finetuned_top10/epoch=epoch=0-val_accuracy=val/accuracy=0.8859.ckpt"),
        #     "results_dir_name": "one_to_one_sliced_data_superglue_finedtuned_top10_best_val_ckpt",
        # }
    ]

    for experiment in experiments:
        checkpoint_path = experiment["checkpoint_path"]
        results_dir_name = experiment["results_dir_name"]

        for parent_root in parent_roots:
            if not parent_root.is_dir():
                print(f"Skipping {parent_root}: not a directory")
                continue

            all_results_core: Dict[str, Dict[str, float]] = {}
            all_results_full: Dict[str, Dict[str, float]] = {}
            overall_ct_totals = _init_celltype_totals(NUM_CELL_TYPES_FINE) if compute_cell_type_metrics else None

            for subgraph_dir in sorted(parent_root.iterdir()):
                if not subgraph_dir.is_dir():
                    continue  # skip stray files

                try:
                    _discover_subgraph_datasets(subgraph_dir)
                except ValueError as exc:
                    if "No dataset folders with a test/ split found" in str(exc):
                        print(f"Skipping {subgraph_dir}: {exc}")
                        continue
                    raise

                results_path = subgraph_dir / results_dir_name / "metrics.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)

                suite_core, suite_full, suite_ct_totals = run_subgraph_suite(
                    subgraph_root=subgraph_dir,
                    checkpoint_path=checkpoint_path,
                    results_path=results_path,
                    compute_cell_type_metrics=compute_cell_type_metrics,
                )

                for dataset_name, metrics in suite_core.items():
                    all_results_core[f"{subgraph_dir.name}/{dataset_name}"] = metrics
                for dataset_name, metrics in suite_full.items():
                    all_results_full[f"{subgraph_dir.name}/{dataset_name}"] = metrics

                if compute_cell_type_metrics and suite_ct_totals is not None:
                    _accumulate_celltype_totals(overall_ct_totals, _celltype_summary_from_totals(suite_ct_totals))

            if all_results_core:
                overall_dir = parent_root / results_dir_name
                overall_dir.mkdir(parents=True, exist_ok=True)

                core_summary = _summarize_metrics(all_results_core)
                full_summary = _summarize_metrics(all_results_full)

                core_path = overall_dir / "overall_metrics_summary_core.json"
                full_path = overall_dir / "overall_metrics_summary_full.json"

                with core_path.open("w") as handle:
                    json.dump(core_summary, handle, indent=2)
                with full_path.open("w") as handle:
                    json.dump(full_summary, handle, indent=2)

                if compute_cell_type_metrics and overall_ct_totals is not None:
                    overall_ct_summary = _celltype_summary_from_totals(overall_ct_totals)
                    with (overall_dir / "overall_celltype_errors_summary.json").open("w") as handle:
                        json.dump(overall_ct_summary, handle, indent=2)
