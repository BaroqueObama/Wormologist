#!/usr/bin/env python
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any, List

import torch
import torch.nn.functional as F

from config.config import Config, TaskConfig
from loader.data_loader import create_data_loader
from loader.coordinate_encoder import CoordinateEncoder
from model.lightning_model import LightningModel
from model.sinkhorn import compute_assignment_accuracy

import json


# ----------------------------
# Loading utilities
# ----------------------------
def handle_state_dict_gpu_mismatch(state_dict: dict, remove_module_prefix: bool = True) -> dict:
    """Handle state dict from multi-GPU training for single GPU or different GPU inference."""
    if not remove_module_prefix:
        return state_dict
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Config = None,
    device: str = "cuda",
    map_location: Optional[str] = None,
    force_task_target: Optional[str] = None,
) -> Tuple[LightningModel, Config]:
    """
    Load a trained model from a checkpoint. Optionally force the task target
    (e.g., "cell_type" for a cell-type classifier or "canonical" for ID prediction).
    """
    if map_location is None:
        map_location = device if device != "cuda" or not torch.cuda.is_available() else None

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if config is None:
        if "hyper_parameters" in checkpoint and "config" in checkpoint["hyper_parameters"]:
            config = checkpoint["hyper_parameters"]["config"]
        else:
            raise ValueError("No config found in checkpoint and none provided")

    if isinstance(config, dict):
        config = Config.from_dict(config)
    if not hasattr(config, "task") or config.task is None:
        config.task = TaskConfig()

    if force_task_target is not None:
        config.task.target = force_task_target

    model = LightningModel(config)
    state_dict = checkpoint["state_dict"]

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        cleaned_state_dict = handle_state_dict_gpu_mismatch(state_dict)
        model.load_state_dict(cleaned_state_dict, strict=False)
        print("Loaded checkpoint with modified state dict (removed module prefix)")

    model = model.to(device)
    model.eval()
    return model, config


# ----------------------------
# Dataset discovery helpers
# ----------------------------
def _discover_subgraph_datasets(subgraph_root: Union[str, Path]) -> List[Path]:
    """
    Find every dataset folder directly under `subgraph_root` that has a `test/` split
    containing at least one HDF5 shard.
    """
    subgraph_root = Path(subgraph_root)
    dataset_dirs: List[Path] = []

    for candidate in sorted(subgraph_root.iterdir()):
        if not candidate.is_dir():
            continue
        test_dir = candidate / "test"
        if test_dir.is_dir() and any(test_dir.glob("*.h5")):
            dataset_dirs.append(candidate)

    # Handle the case where subgraph_root itself is a dataset folder
    if (subgraph_root / "test").is_dir() and any((subgraph_root / "test").glob("*.h5")):
        if subgraph_root not in dataset_dirs:
            dataset_dirs.insert(0, subgraph_root)

    if not dataset_dirs:
        raise ValueError(f"No dataset folders with a test/ split found under {subgraph_root}")
    return dataset_dirs


# ----------------------------
# Hierarchical helpers
# ----------------------------
def _to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to"):          # handles torch_geometric Batch/Data
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def _expected_ct_width(canon_model: LightningModel, canon_cfg: Config) -> int:
    coord_dim = CoordinateEncoder.get_output_dim(canon_cfg.data.coordinate_system)
    pe_dim = (
        canon_cfg.model.num_scales * canon_cfg.model.k_hops
        if canon_cfg.model.use_multiscale_pe
        else canon_cfg.model.k_hops
    )
    return canon_model.model.node_encoder.in_features - (coord_dim + pe_dim)


def _append_predicted_celltypes(
    batch_device: Dict[str, Any],
    ct_model: LightningModel,
    ct_feat_mode: str,
    expected_ct_width: int,
) -> None:
    """
    Mutates batch_device["batch"].x by concatenating predicted cell-type features.

    ct_feat_mode: "soft" -> probabilities, "hard" -> one-hot argmax
    """
    ct_out = ct_model(batch_device)
    ct_logits = ct_out["logits"]  # [B, max_nodes, C]
    ct_classes = ct_logits.shape[-1]

    if ct_classes != expected_ct_width:
        raise ValueError(
            f"Canonical model expects {expected_ct_width} cell-type features, "
            f"but cell-type model outputs {ct_classes}"
        )

    if ct_feat_mode == "soft":
        ct_feat = torch.softmax(ct_logits, dim=-1)
    else:
        ct_feat = F.one_hot(ct_logits.argmax(dim=-1), num_classes=ct_classes).float()

    vis_mask = batch_device["visible_mask"]  # [B, max_nodes]
    per_node_feat = []
    for b in range(vis_mask.shape[0]):
        n = int(vis_mask[b].sum())
        if n > 0:
            per_node_feat.append(ct_feat[b, :n])
    if per_node_feat:
        ct_concat = torch.cat(per_node_feat, dim=0)
        pyg_batch = batch_device["batch"]
        pyg_batch.x = torch.cat([pyg_batch.x, ct_concat], dim=-1)


def _init_totals() -> Dict[str, float]:
    return {
        "exact_accuracy_sum": 0.0,
        "exact_accuracy_sum_sq": 0.0,
        "exact_accuracy_count": 0.0,
        "top3_accuracy_sum": 0.0,
        "top3_accuracy_sum_sq": 0.0,
        "top3_accuracy_count": 0.0,
        "top5_accuracy_sum": 0.0,
        "top5_accuracy_sum_sq": 0.0,
        "top5_accuracy_count": 0.0,
        "total_nodes": 0.0,
        "num_worms": 0.0,
        "hungarian_sum": 0.0,
        "hungarian_count": 0.0,
        "logits_top1_sum": 0.0,
        "logits_top3_sum": 0.0,
        "logits_top5_sum": 0.0,
        "logits_count": 0.0,
    }


def _accumulate_totals(totals: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key in ("exact_accuracy", "top3_accuracy", "top5_accuracy"):
        totals[f"{key}_sum"] += float(metrics.get(f"{key}_sum", 0.0))
        totals[f"{key}_sum_sq"] += float(metrics.get(f"{key}_sum_sq", 0.0))
        totals[f"{key}_count"] += float(metrics.get(f"{key}_count", 0.0))

    totals["total_nodes"] += float(metrics.get("total_nodes", 0.0))
    num_worms = float(metrics.get("num_worms", 0.0))
    totals["num_worms"] += num_worms

    if "exact_accuracy_hungarian" in metrics:
        totals["hungarian_sum"] += float(metrics["exact_accuracy_hungarian"]) * num_worms
        totals["hungarian_count"] += num_worms

    if "exact_accuracy_logits" in metrics:
        totals["logits_top1_sum"] += float(metrics["exact_accuracy_logits"]) * num_worms
        totals["logits_top3_sum"] += float(metrics.get("top3_accuracy_logits", 0.0)) * num_worms
        totals["logits_top5_sum"] += float(metrics.get("top5_accuracy_logits", 0.0)) * num_worms
        totals["logits_count"] += num_worms


def _finalize_totals(totals: Dict[str, float]) -> Dict[str, float]:
    def _avg(sum_key: str, count_key: str) -> float:
        return totals[sum_key] / totals[count_key] if totals[count_key] else 0.0

    result = {
        "exact_accuracy": _avg("exact_accuracy_sum", "exact_accuracy_count"),
        "top3_accuracy": _avg("top3_accuracy_sum", "top3_accuracy_count"),
        "top5_accuracy": _avg("top5_accuracy_sum", "top5_accuracy_count"),
        "total_nodes": int(totals["total_nodes"]),
        "num_worms": int(totals["num_worms"]),
    }

    if totals["hungarian_count"]:
        result["exact_accuracy_hungarian"] = totals["hungarian_sum"] / totals["hungarian_count"]
    if totals["logits_count"]:
        result["exact_accuracy_logits"] = totals["logits_top1_sum"] / totals["logits_count"]
        result["top3_accuracy_logits"] = totals["logits_top3_sum"] / totals["logits_count"]
        result["top5_accuracy_logits"] = totals["logits_top5_sum"] / totals["logits_count"]
    return result


def _summarize_suite(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean/std across dataset splits for the main metrics."""
    summary: Dict[str, float] = {}
    if not results:
        return summary
    keys = set().union(*(m.keys() for m in results.values()))
    for k in keys:
        vals = [float(m[k]) for m in results.values() if k in m and isinstance(m[k], (int, float))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        summary[f"{k}/mean"] = mean
        summary[f"{k}/std"] = std
    return summary


# ----------------------------
# Core hierarchical inference
# ----------------------------
def _hierarchical_batch_step(
    batch: Dict[str, Any],
    canonical_model: LightningModel,
    celltype_model: LightningModel,
    expected_ct_width: int,
    ct_feat_mode: str = "hard",
    compute_hungarian: bool = False,
) -> Dict[str, float]:
    """
    One forward pass that:
      1) predicts cell types,
      2) appends them as features,
      3) runs canonical model + matcher,
      4) returns canonical metrics.
    """
    _append_predicted_celltypes(batch, celltype_model, ct_feat_mode, expected_ct_width)
    outputs = canonical_model(batch)
    pyg_batch = batch["batch"]
    labels = pyg_batch.y
    batch_assignment = pyg_batch.batch
    visible_mask = batch["visible_mask"]
    targets = canonical_model.prepare_targets(labels, batch_assignment, visible_mask)
    loss_dict = canonical_model.loss_fn(outputs, targets)
    metrics = compute_assignment_accuracy(
        outputs,
        targets,
        loss_dict["soft_assignments"],
        dustbin_weights=loss_dict.get("dustbin_weights", None),
        dustbin_row_weights=loss_dict.get("dustbin_row_weights", None),
        compute_hungarian=compute_hungarian,
        full_assignments=loss_dict.get("full_assignments", None),
    )
    return metrics


def run_hierarchical_subgraph_suite(
    subgraph_root: Union[str, Path],
    canonical_ckpt: Union[str, Path],
    celltype_ckpt: Union[str, Path],
    use_soft_celltypes: bool = False,
    compute_hungarian: bool = False,
    results_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate canonical-ID accuracy using predicted cell-type features from a separate model.
    Ground-truth cell types are never used as inputs.

    Returns per-dataset metrics; optionally writes JSON if results_path is provided.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    canon_model, canon_cfg = load_model_from_checkpoint(canonical_ckpt, device=device, force_task_target="canonical")
    ct_model, _ = load_model_from_checkpoint(celltype_ckpt, device=device, force_task_target="cell_type")

    ct_feat_mode = "soft" if use_soft_celltypes else "hard"
    expected_ct_width = _expected_ct_width(canon_model, canon_cfg)

    results: Dict[str, Dict[str, float]] = {}

    for dataset_dir in _discover_subgraph_datasets(subgraph_root):
        loader = create_data_loader(
            data_path=dataset_dir,
            split="test",
            batch_size=canon_cfg.training.micro_batch_size,
            coordinate_system=canon_cfg.data.coordinate_system,
            normalize_coords=canon_cfg.data.normalize_coords,
            use_cell_type_features=False,  # critical: do not leak GT cell types
            augmentation_config=None,
            num_workers=0,
            distributed=False,
            shuffle_shards=False,
            shuffle_within_shard=False,
            target="canonical",
        )

        totals = _init_totals()
        with torch.no_grad():
            for batch in loader:
                batch_device = _to_device(batch, device)
                metrics = _hierarchical_batch_step(
                    batch_device,
                    canon_model,
                    ct_model,
                    expected_ct_width=expected_ct_width,
                    ct_feat_mode=ct_feat_mode,
                    compute_hungarian=compute_hungarian,
                )
                _accumulate_totals(totals, metrics)

        results[dataset_dir.name] = _finalize_totals(totals)
        print(f"{dataset_dir.name}: {results[dataset_dir.name]}")

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2))
        summary = _summarize_suite(results)
        results_path.with_name(f"{results_path.stem}_summary{results_path.suffix}").write_text(
            json.dumps(summary, indent=2)
        )

    return results


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":

    parent_root = Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/real_test_set_presentation/40_slices")
    canonical_ckpt = Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_all_cell_types_superglue_loss_wo_trainable_dustbins/all_cell_types_sliced_data_superglue_loss_wo_trainable_dustbins/epoch=epoch=0-val_accuracy=val/accuracy=0.9133.ckpt")  # <-- set me
    celltype_ckpt = Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_cell_types_prediction/sliced_data_cell_type_prediction/epoch=epoch=0-val_accuracy=val/accuracy=0.9195.ckpt")    # <-- set me

    use_soft_celltypes = True   # True => pass soft probs instead of argmax one-hot
    compute_hungarian = True   # True => also compute Hungarian accuracy (slower)
    results_root = Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/real_test_set_presentation/40_slices/hierarchical_metrics_soft")         # e.g., Path("/tmp/hierarchical_metrics.json"); None to skip writes

    all_results = {}
    for subgraph_dir in sorted(parent_root.iterdir()):
        if not subgraph_dir.is_dir():
            continue
        try:
            suite_results = run_hierarchical_subgraph_suite(
                subgraph_root=subgraph_dir,
                canonical_ckpt=canonical_ckpt,
                celltype_ckpt=celltype_ckpt,
                use_soft_celltypes=use_soft_celltypes,
                compute_hungarian=compute_hungarian,
                results_path=(
                    (results_root / f"{subgraph_dir.name}_metrics.json")
                    if results_root is not None else None
                ),
            )
        except ValueError as exc:
            print(f"Skipping {subgraph_dir}: {exc}")
            continue

        for name, metrics in suite_results.items():
            all_results[f"{subgraph_dir.name}/{name}"] = metrics

    print("Summary across all datasets:")
    print(json.dumps(_summarize_suite(all_results), indent=2))

