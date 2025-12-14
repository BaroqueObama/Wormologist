import torch
import numpy as np
import random
from typing import Tuple, Optional
from scipy import stats
from loader.slicing import get_slice_indices


class CurriculumNodeDropper:
    """Handles curriculum learning with progressive node dropping."""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.enabled
        
        random.seed(config.seed)
        np.random.seed(config.seed)

        self.strategy = getattr(config, "subgraph_strategy", "random")
        slice_seed = getattr(config, "slice_seed", config.seed if hasattr(config, "seed") else 42)
        self.slice_rng = np.random.default_rng(slice_seed)
        self.slice_max_n_slices = getattr(config, "slice_max_n_slices", 40)
        self.slice_profiles = getattr(config, "slice_profiles", [])
        
        
    def get_visibility_rate(self, global_batch: int, split: str) -> float:
        if not self.enabled:
            return 1.0
        
        if split in ["val", "test"]:
            return self._sample_val_visibility()
        
        return self._sample_train_visibility(global_batch)
    
    
    def _sample_train_visibility(self, global_batch: int) -> float:
        phase = self.config.get_phase(global_batch)
        target = self.config.get_curriculum_target(global_batch)
        
        if phase == "warmup":
            return 1.0
        
        elif phase == "curriculum":
            return self._sample_from_distribution(min_val=self.config.end_visibility, max_val=1.0, target=target, phase=phase)
        
        elif phase == "cooldown":
            return self._sample_from_distribution(min_val=self.config.end_visibility, max_val=1.0, target=self.config.end_visibility, phase=phase)
        
        else:
            return np.random.uniform(self.config.end_visibility, self.config.start_visibility)


    def _sample_from_distribution(self, min_val: float, max_val: float, target: float, phase: str = None) -> float:
        if self.config.train_distribution == "beta":
            if phase == "cooldown":
                alpha = self.config.cooldown_beta_alpha
                beta = self.config.cooldown_beta_beta
            else:
                # Fallback to curriculum parameters
                alpha = self.config.curriculum_beta_alpha
                beta = self.config.curriculum_beta_beta
            
            beta_sample = np.random.beta(alpha, beta)
            visibility = target + beta_sample * (max_val - target)
            
        elif self.config.train_distribution == "truncated_normal":
            mean = (target + max_val) / 2
            std = self.config.truncated_std * (max_val - target)
            a = (target - mean) / std
            b = (max_val - mean) / std
            
            sample = stats.truncnorm.rvs(a, b, loc=mean, scale=std)
            visibility = float(sample)
            
        else:
            visibility = np.random.uniform(target, max_val)
        
        return np.clip(visibility, min_val, max_val)
    
    
    def _sample_val_visibility(self) -> float:
        return np.random.uniform(self.config.val_min_visibility, self.config.val_max_visibility)

    def drop_nodes(self, coords: torch.Tensor, labels: torch.Tensor, visibility_rate: float, global_batch: int, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly drop nodes to achieve target visibility."""
        
        if self.strategy == "sliced":
            n_slices = self._visibility_to_n_slices(visibility_rate, split)
            return self._drop_nodes_sliced(coords, labels, n_slices, visibility_rate)

        n_nodes = coords.shape[0]
        n_keep = max(1, int(n_nodes * visibility_rate))
        
        perm = torch.randperm(n_nodes, device=coords.device)
        keep_indices = perm[:n_keep].sort()[0]  # TODO: Remove this sort to make node order random?
        
        dropped_coords = coords[keep_indices]
        dropped_labels = labels[keep_indices]
        
        return dropped_coords, dropped_labels, keep_indices
    
    def _sample_slice_params(self, n_slices: int, ap_span: float, visibility: float,) -> Tuple[float, float, float, Optional[str], str]:

        if not self.slice_profiles:
            raise RuntimeError("slice_profiles is empty; please define at least one profile.")

        profile = self.slice_profiles[self.slice_rng.integers(len(self.slice_profiles))]

        if "thickness_fraction" in profile:
            thickness = float(profile["thickness_fraction"]) * ap_span
        else:
            thickness = float(profile.get("slice_thickness", ap_span / max(n_slices, 1)))

        base_range = profile.get("crop_fraction_range", [self.config.slice_crop_fraction, self.config.slice_crop_fraction],)
        
        base_low, base_high = base_range
        base_low = max(0.0, min(1.0, float(base_low)))
        base_high = max(0.0, min(1.0, float(base_high)))

        if getattr(self.config, "interpolate_crop_fraction", False):
            start_vis = getattr(self.config, "start_visibility", 1.0)
            end_vis = getattr(self.config, "end_visibility", start_vis)
            v = max(min(visibility, start_vis), end_vis)
            denom = max(start_vis - end_vis, 1e-6)
            progress = (start_vis - v) / denom

            scaled_low = 1.0 - progress * (1.0 - base_low)
            scaled_high = 1.0 - progress * (1.0 - base_high)
            scaled_low, scaled_high = min(scaled_low, scaled_high), max(scaled_low, scaled_high)
        else:
            scaled_low, scaled_high = base_low, base_high

        crop_fraction = float(self.slice_rng.uniform(scaled_low, scaled_high))
        crop_axis = profile.get("crop_axis", self.config.slice_crop_axis)

        if n_slices > 1:
            center_spacing = ap_span / (n_slices - 1)
            max_shift = max(0.0, center_spacing - thickness)
        else:
            max_shift = 0.0
        shift = float(self.slice_rng.uniform(0.0, max_shift))

        crop_side = "negative" if self.slice_rng.random() < 0.5 else "positive"

        return thickness, shift, crop_fraction, crop_axis, crop_side



    
    def _visibility_to_n_slices(self, visibility: float, split: str) -> int:
        if self.config.subgraph_strategy != "sliced":
            return 0

        # Left in case we want to have a different behavior for val/test
        if split != "train":
            return max(1, int(round(self.slice_max_n_slices * visibility)))

        return max(1, int(round(self.slice_max_n_slices * visibility)))

    
    def _drop_nodes_sliced(self, coords: torch.Tensor, labels: torch.Tensor, n_slices: int, visibility: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if n_slices <= 0:
            keep = torch.arange(coords.size(0), device=coords.device)
            return coords, labels, keep

        points = coords.cpu().numpy()
        z_min = float(points[:, 2].min())
        z_max = float(points[:, 2].max())
        ap_span = max(z_max - z_min, 1e-6)

        thickness, shift, crop_fraction, crop_axis, crop_side = self._sample_slice_params(n_slices, ap_span, visibility)

        rng = self.slice_rng if (crop_axis == "random" or self.config.slice_crop_axis == "random") else None

        indices, _, projected_coords = get_slice_indices(
            points,
            n_slices=n_slices,
            slice_thickness=thickness,
            shift=shift,
            crop_axis=crop_axis,
            crop_side=crop_side,
            crop_fraction=crop_fraction,
            random_state=rng,
        )

        if indices.size == 0:
            keep = torch.arange(coords.size(0), device=coords.device)
            projected = coords[keep]
        else:
            keep = torch.from_numpy(indices).to(coords.device)
            projected = torch.from_numpy(projected_coords).to(coords.device, coords.dtype)

        return projected, labels[keep], keep

    def get_curriculum_info(self, global_batch: int) -> dict:
        """Get information about current curriculum state."""
        
        phase = self.config.get_phase(global_batch)
        target = self.config.get_curriculum_target(global_batch)
        
        return {
            'phase': phase,
            'target_visibility': target,
            'global_batch': global_batch,
            'enabled': self.enabled
        }
