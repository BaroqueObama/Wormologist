#!/usr/bin/env python
from pathlib import Path
import json
from typing import Dict, Optional, Union, Tuple, Any

import torch
import pytorch_lightning as L

from config.config import Config, TaskConfig
from loader.data_loader import create_data_loader
from model.lightning_model import LightningModel

import numpy as np
from loader.cell_types import NUM_CELL_TYPES_FINE

# Keep only the metrics the cell-type task logs in test_step
METRICS_TO_SAVE = {
    "test/loss",
    "test/accuracy",
    "test/top3_accuracy",
    "test/top5_accuracy",
}

def compute_cell_type_error_stats_for_loader(
    model: LightningModel,
    loader,
    device: str = "cuda",
    num_cell_types: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    n_classes = num_cell_types or getattr(getattr(model, "config", None), "task", TaskConfig()).num_classes

    total_per_type = np.zeros(n_classes, dtype=np.int64)
    correct_per_type = np.zeros(n_classes, dtype=np.int64)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            pyg_batch = batch_device["batch"]
            labels = pyg_batch.y.long()
            batch_assignment = pyg_batch.batch.long()
            visible_mask = batch_device["visible_mask"]

            targets = model.prepare_targets(labels, batch_assignment, visible_mask)
            outputs = model(batch_device)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)

            labels_cpu = targets["labels"].cpu()
            visible_mask_cpu = targets["visible_mask"].cpu()
            preds_cpu = preds.cpu()

            B, _ = preds_cpu.shape
            for b in range(B):
                vis = visible_mask_cpu[b]
                if not vis.any():
                    continue
                true_b = labels_cpu[b, vis].numpy()
                pred_b = preds_cpu[b, vis].numpy()

                for ct in range(n_classes):
                    mask_ct = (true_b == ct)
                    if mask_ct.any():
                        total_ct = int(mask_ct.sum())
                        total_per_type[ct] += total_ct
                        correct_per_type[ct] += int((mask_ct & (pred_b == true_b)).sum())

                mis_mask = (true_b != pred_b)
                if mis_mask.any():
                    for t, p in zip(true_b[mis_mask], pred_b[mis_mask]):
                        confusion[int(t), int(p)] += 1

    total_nodes = int(total_per_type.sum())
    total_mispreds = int(confusion.sum())

    per_type_summary: Dict[str, Any] = {}
    for ct in range(n_classes):
        total_ct = int(total_per_type[ct])
        correct_ct = int(correct_per_type[ct])
        mis_ct = total_ct - correct_ct
        per_type_summary[str(ct)] = {
            "total_nodes": total_ct,
            "correct": correct_ct,
            "mispredicted": mis_ct,
            "accuracy": (correct_ct / total_ct) if total_ct > 0 else 0.0,
            "fraction_of_all_nodes": (total_ct / total_nodes) if total_nodes > 0 else 0.0,
            "fraction_of_all_errors": (mis_ct / total_mispreds) if total_mispreds > 0 else 0.0,
        }

    row_norm_conf = np.zeros_like(confusion, dtype=np.float64)
    for ct in range(n_classes):
        row_sum = confusion[ct].sum()
        if row_sum > 0:
            row_norm_conf[ct] = confusion[ct] / row_sum

    return {
        "num_cell_types": n_classes,
        "per_cell_type": per_type_summary,
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_row_normalized": row_norm_conf.tolist(),
        "total_mispredictions": total_mispreds,
        "total_nodes": total_nodes,
    }

def _init_celltype_totals(num_cell_types: int = NUM_CELL_TYPES_FINE) -> Dict[str, Any]:
    return {
        "total_per_type": np.zeros(num_cell_types, dtype=np.int64),
        "correct_per_type": np.zeros(num_cell_types, dtype=np.int64),
        "confusion": np.zeros((num_cell_types, num_cell_types), dtype=np.int64),
        "total_nodes": 0,
    }

def _accumulate_celltype_totals(totals: Dict[str, Any], ct_stats: Dict[str, Any]) -> None:
    num_cell_types = len(ct_stats["per_cell_type"])
    per = ct_stats["per_cell_type"]
    totals["total_per_type"] += np.array([per[str(i)]["total_nodes"] for i in range(num_cell_types)], dtype=np.int64)
    totals["correct_per_type"] += np.array([per[str(i)]["correct"] for i in range(num_cell_types)], dtype=np.int64)
    totals["confusion"] += np.array(ct_stats["confusion_matrix_counts"], dtype=np.int64)
    totals["total_nodes"] += ct_stats["total_nodes"]


def _celltype_summary_from_totals(totals: Dict[str, Any]) -> Dict[str, Any]:
    num_cell_types = totals["total_per_type"].shape[0]
    total_nodes = int(totals["total_per_type"].sum())
    correct_total = int(totals["correct_per_type"].sum())
    total_mispreds = total_nodes - correct_total


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

    return {
        "num_cell_types": num_cell_types,
        "per_cell_type": per_type_summary,
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_row_normalized": row_norm_conf.tolist(),
        "total_mispredictions": total_mispreds,
        "total_nodes": total_nodes,
    }

def _filter_metrics(metrics: Dict[str, float], keep: set[str]) -> Dict[str, float]:
    if not keep:
        return metrics
    return {k: v for k, v in metrics.items() if k in keep}

def _discover_subgraph_datasets(subgraph_root: Union[str, Path]) -> list[Path]:
    root = Path(subgraph_root)
    dataset_dirs: list[Path] = []
    if (root / "test").glob("*.h5"):
        if any((root / "test").glob("*.h5")):
            dataset_dirs.append(root)
    for cand in sorted(root.iterdir()):
        if not cand.is_dir():
            continue
        test_dir = cand / "test"
        if any(test_dir.glob("*.h5")):
            dataset_dirs.append(cand)
    if not dataset_dirs:
        raise ValueError(f"No dataset folders with a test/ split under {subgraph_root}")
    return dataset_dirs

def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "cuda",
    map_location: Optional[str] = None,
) -> Tuple[LightningModel, Config]:
    
    if map_location is None:
        map_location = device if device != "cuda" or not torch.cuda.is_available() else None

    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    cfg = ckpt["hyper_parameters"]["config"]
    if isinstance(cfg, dict):
        cfg = Config.from_dict(cfg)
    if not hasattr(cfg, "task") or cfg.task is None:
        cfg.task = TaskConfig()
    cfg.task.target = "cell_type"

    model = LightningModel(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg

def run_subgraph_suite(
    subgraph_root: Union[str, Path],
    checkpoint_path: Union[str, Path],
    results_path: Optional[Union[str, Path]] = None,
    use_cell_type_features: bool = False,  # set True if your model was trained with these inputs
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

    results_file = None
    if results_path is not None:
        results_file = Path(results_path)
        if results_file.is_dir():
            results_file = results_file / "subgraph_metrics.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

    results_core: Dict[str, Dict[str, float]] = {}
    results_full: Dict[str, Dict[str, float]] = {}

    num_cell_types = getattr(config.task, "num_classes", NUM_CELL_TYPES_FINE)
    ct_totals = _init_celltype_totals(num_cell_types) if compute_cell_type_metrics else None

    for dataset_dir in _discover_subgraph_datasets(subgraph_root):
        loader = create_data_loader(
            data_path=dataset_dir,
            split="test",
            batch_size=config.training.micro_batch_size,
            coordinate_system=config.data.coordinate_system,
            normalize_coords=config.data.normalize_coords,
            use_cell_type_features=use_cell_type_features,
            augmentation_config=None,
            num_workers=0,
            distributed=False,
            shuffle_shards=False,
            shuffle_within_shard=False,
            target="cell_type",
        )
        metrics = trainer.test(model, dataloaders=loader, ckpt_path=None, verbose=False)
        split_metrics = metrics[0] if metrics else {}
        results_full[dataset_dir.name] = split_metrics
        results_core[dataset_dir.name] = _filter_metrics(split_metrics, METRICS_TO_SAVE)
        print(f"{dataset_dir.name}: {results_core[dataset_dir.name]}")

        if compute_cell_type_metrics:
            ct_stats = compute_cell_type_error_stats_for_loader(
                model,
                loader,
                device=device,
                num_cell_types=num_cell_types,
            )
            if results_file is not None:
                ct_path = results_file.with_name(f"{dataset_dir.name}_celltype_errors.json")
                with ct_path.open("w") as h:
                    json.dump(ct_stats, h, indent=2)
            _accumulate_celltype_totals(ct_totals, ct_stats)

        if results_file is not None:
            core_path = results_file.with_name(f"{results_file.stem}_core{results_file.suffix}")
            full_path = results_file.with_name(f"{results_file.stem}_full{results_file.suffix}")
            core_path.parent.mkdir(parents=True, exist_ok=True)
            with core_path.open("w") as h:
                json.dump(results_core, h, indent=2)
            with full_path.open("w") as h:
                json.dump(results_full, h, indent=2)

    if results_file is not None and results_core:
        def _summarize(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            if not results:
                return out
            metric_keys = set().union(*(m.keys() for m in results.values()))
            for k in metric_keys:
                vals = [float(m[k]) for m in results.values() if isinstance(m.get(k), (int, float))]
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                if len(vals) > 1:
                    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                    std = var ** 0.5
                else:
                    std = 0.0
                out[f"{k}/mean"] = mean
                out[f"{k}/std"] = std
            return out

        core_summary = _summarize(results_core)
        full_summary = _summarize(results_full)
        with results_file.with_name("metrics_summary_core.json").open("w") as h:
            json.dump(core_summary, h, indent=2)
        with results_file.with_name("metrics_summary_full.json").open("w") as h:
            json.dump(full_summary, h, indent=2)

        if compute_cell_type_metrics and ct_totals is not None:
            ct_summary = _celltype_summary_from_totals(ct_totals)
            with results_file.with_name("celltype_errors_summary.json").open("w") as h:
                json.dump(ct_summary, h, indent=2)

    return results_core, results_full, ct_totals

if __name__ == "__main__":
    compute_cell_type_features = False
    compute_cell_type_metrics = True  # set False to skip error stats
    parent_roots = [
        Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/real_test_set_presentation/20_slices"),
        Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/real_test_set_presentation/40_slices"),
        # Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/random_comparison_to_real_test_presentation/20_slices"),
        # Path("/fs/pool/pool-mlsb/bulat/Wormologist/presentation_results/random_comparison_to_real_test_presentation/40_slices"),
    ]
    experiments = [
        {
            "checkpoint_path": Path("/fs/pool/pool-mlsb/bulat/Wormologist/model_checkpoints/checkpoint_cell_types_prediction/sliced_data_cell_type_prediction/epoch=epoch=0-val_accuracy=val/accuracy=0.9195.ckpt"),
            "results_dir_name": "cell_type_prediction_best_val_ckpt_with_cell_type_stats",
        },
    ]

    for experiment in experiments:
        checkpoint_path = experiment["checkpoint_path"]
        results_dir_name = experiment["results_dir_name"]
        for parent_root in parent_roots:
            if not parent_root.is_dir():
                print(f"Skipping {parent_root}: not a directory")
                continue
            all_core: Dict[str, Dict[str, float]] = {}
            all_full: Dict[str, Dict[str, float]] = {}
            overall_ct_totals: Optional[Dict[str, Any]] = None
            for subgraph_dir in sorted(parent_root.iterdir()):
                if not subgraph_dir.is_dir():
                    continue
                try:
                    _discover_subgraph_datasets(subgraph_dir)
                except ValueError as exc:
                    print(f"Skipping {subgraph_dir}: {exc}")
                    continue
                results_path = subgraph_dir / results_dir_name / "metrics.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                suite_core, suite_full, suite_ct_totals = run_subgraph_suite(
                    subgraph_root=subgraph_dir,
                    checkpoint_path=checkpoint_path,
                    results_path=results_path,
                    use_cell_type_features=compute_cell_type_features,
                    compute_cell_type_metrics=compute_cell_type_metrics,
                )
                for name, metrics in suite_core.items():
                    all_core[f"{subgraph_dir.name}/{name}"] = metrics
                for name, metrics in suite_full.items():
                    all_full[f"{subgraph_dir.name}/{name}"] = metrics
                if compute_cell_type_metrics and suite_ct_totals is not None:
                    if overall_ct_totals is None:
                        overall_ct_totals = _init_celltype_totals(suite_ct_totals["total_per_type"].shape[0])
                    _accumulate_celltype_totals(overall_ct_totals, _celltype_summary_from_totals(suite_ct_totals))
            if all_core:
                overall_dir = parent_root / results_dir_name
                overall_dir.mkdir(parents=True, exist_ok=True)
                def _summarize(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
                    out: Dict[str, float] = {}
                    if not results:
                        return out
                    metric_keys = set().union(*(m.keys() for m in results.values()))
                    for k in metric_keys:
                        vals = [float(m[k]) for m in results.values() if isinstance(m.get(k), (int, float))]
                        if not vals:
                            continue
                        mean = sum(vals) / len(vals)
                        std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
                        out[f"{k}/mean"] = mean
                        out[f"{k}/std"] = std
                    return out
                with (overall_dir / "overall_metrics_summary_core.json").open("w") as h:
                    json.dump(_summarize(all_core), h, indent=2)
                with (overall_dir / "overall_metrics_summary_full.json").open("w") as h:
                    json.dump(_summarize(all_full), h, indent=2)
                if compute_cell_type_metrics and overall_ct_totals is not None:
                    overall_ct_summary = _celltype_summary_from_totals(overall_ct_totals)
                    with (overall_dir / "overall_celltype_errors_summary.json").open("w") as h:
                        json.dump(overall_ct_summary, h, indent=2)
