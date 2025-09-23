import torch
import numpy as np
import random
from typing import Tuple
from scipy import stats


class CurriculumNodeDropper:
    """Handles curriculum learning with progressive node dropping."""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.enabled
        
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        
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
    
    
    def drop_nodes(self, coords: torch.Tensor, labels: torch.Tensor, visibility_rate: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly drop nodes to achieve target visibility."""
        
        n_nodes = coords.shape[0]
        n_keep = max(1, int(n_nodes * visibility_rate))
        
        perm = torch.randperm(n_nodes, device=coords.device)
        keep_indices = perm[:n_keep].sort()[0]  # TODO: Remove this sort to make node order random?
        
        dropped_coords = coords[keep_indices]
        dropped_labels = labels[keep_indices]
        
        return dropped_coords, dropped_labels, keep_indices

    
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
