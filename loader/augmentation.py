import torch
import numpy as np


class DataAugmentation:
    """
    Data augmentation order:
    1. Z-axis normalization (min z -> 0)
    2. Z-axis shift (random translation)
    3. Random uniform scaling
    4. Z-axis rotation
    5. Post-rotation x-y scaling (asymmetric deformation)
    """
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.enabled
        self.normalize_z_axis_enabled = getattr(config, "normalize_z_axis", True)
        
        # Z-shift parameters
        self.z_shift_range = config.z_shift_range
        self.z_shift_distribution = config.z_shift_distribution
        self.z_shift_beta_alpha = config.z_shift_beta_alpha
        self.z_shift_beta_beta = config.z_shift_beta_beta
        
        # Uniform scaling parameters
        self.uniform_scale_range = config.uniform_scale_range
        self.uniform_scale_distribution = config.uniform_scale_distribution
        self.uniform_scale_beta_alpha = config.uniform_scale_beta_alpha
        self.uniform_scale_beta_beta = config.uniform_scale_beta_beta
        
        # Rotation parameters
        self.z_rotation_range = config.z_rotation_range
        self.rotation_distribution = config.rotation_distribution
        self.rotation_beta_alpha = config.rotation_beta_alpha
        self.rotation_beta_beta = config.rotation_beta_beta
        
        # Post-rotation xy-scale parameters
        self.post_rotation_xy_scale_range = config.post_rotation_xy_scale_range
        self.post_rotation_xy_scale_distribution = config.post_rotation_xy_scale_distribution
        self.post_rotation_xy_scale_beta_alpha = config.post_rotation_xy_scale_beta_alpha
        self.post_rotation_xy_scale_beta_beta = config.post_rotation_xy_scale_beta_beta
        self.xy_scale_orientation_range = config.xy_scale_orientation_range

        # XY jitter parameters (optional planar translation)
        self.xy_jitter_mode = getattr(config, "xy_jitter_mode", "none")  # "none", "normal", or "uniform"
        self.xy_jitter_std = getattr(config, "xy_jitter_std", 0.0)       # used when mode == "normal"
        self.xy_jitter_range = getattr(config, "xy_jitter_range", 0.0)   # half-width when mode == "uniform"

        # Control parameters
        self.apply_to_splits = config.apply_to_splits
        self.base_seed = config.seed
    
    
    def should_augment(self, split: str) -> bool:
        return self.enabled and split in self.apply_to_splits
    
    
    def _get_generator(self, specimen_idx: int) -> torch.Generator:
        generator = torch.Generator()
        seed = self.base_seed + specimen_idx
        generator.manual_seed(seed)
        return generator


    def _sample_beta(self, alpha: float, beta: float, min_val: float, max_val: float, generator: torch.Generator) -> float:
        """Sample from a beta distribution mapped to a specific range."""
        seed_tensor = torch.randint(0, 2**32, (1,), generator=generator)
        np.random.seed(seed_tensor.item())
        
        beta_sample = np.random.beta(alpha, beta)
        
        return min_val + beta_sample * (max_val - min_val)
    
    
    def _sample_uniform(self, min_val: float, max_val: float, generator: torch.Generator) -> float:
        return torch.rand(1, generator=generator).item() * (max_val - min_val) + min_val
    
    
    def normalize_z_axis(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates so that minimum z-value is 0."""
        min_z = coords[:, 2].min()
        coords = coords.clone()
        coords[:, 2] -= min_z
        return coords
    
    
    def apply_z_shift(self, coords: torch.Tensor, z_shift: float) -> torch.Tensor:
        """Apply upward shift along z-axis to account for distance between head nuclei center and top of head."""
        coords = coords.clone()
        coords[:, 2] += z_shift
        return coords
    
    
    def apply_uniform_scaling(self, coords: torch.Tensor, scale: float) -> torch.Tensor:
        return coords * scale
    
    
    def apply_z_rotation(self, coords: torch.Tensor, angle_rad: float) -> torch.Tensor:
        """Apply rotation around z-axis (rotates around origin without centering)."""

        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ], dtype=coords.dtype, device=coords.device)
        
        rotated_coords = torch.matmul(coords, rotation_matrix.T)
        return rotated_coords
    
    
    def apply_post_rotation_xy_scaling(self, coords: torch.Tensor, scale: float, orientation_rad: float) -> torch.Tensor:
        """Apply asymmetric scaling in the xy-plane by "scale" and "1/scale" orthogonally along a random orientation."""
        
        cos_orient = np.cos(orientation_rad)
        sin_orient = np.sin(orientation_rad)
        
        R = torch.tensor([
            [cos_orient, -sin_orient],
            [sin_orient, cos_orient]
        ], dtype=coords.dtype, device=coords.device)
        R_inv = R.T
        
        xy_coords = coords[:, :2]
        z_coords = coords[:, 2:3]
        xy_rotated = torch.matmul(xy_coords, R.T)
        
        scale_matrix = torch.tensor([
            [scale, 0],
            [0, 1.0 / scale]
        ], dtype=coords.dtype, device=coords.device)
        
        xy_scaled = torch.matmul(xy_rotated, scale_matrix.T)
        xy_final = torch.matmul(xy_scaled, R_inv.T)
        result = torch.cat([xy_final, z_coords], dim=1)
        return result
    
    def apply_xy_jitter(self, coords: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Apply a small translation in the xy plane; z is untouched.

        The configured std/range are interpreted as fractions of the current xy span.
        """

        xy_max = coords[:, :2].max(dim=0).values
        xy_min = coords[:, :2].min(dim=0).values
        xy_span = (xy_max - xy_min).clamp_min(1e-6)

        if self.xy_jitter_mode == "normal" and self.xy_jitter_std > 0:
            scale = self.xy_jitter_std * xy_span
            noise = torch.randn(
                (coords.size(0), 2),
                generator=generator,
                device=coords.device,
                dtype=coords.dtype,
            ) * scale
        elif self.xy_jitter_mode == "uniform" and self.xy_jitter_range > 0:
            half_width = self.xy_jitter_range * xy_span
            noise = (
                torch.empty(
                    coords.size(0), 2,
                    device=coords.device,
                    dtype=coords.dtype,
                    generator=generator,
                ).uniform_(-1.0, 1.0)
                * half_width
            )
        else:
            return coords

        out = coords.clone()
        out[:, :2] += noise
        return out
    
    def augment(self, coords: torch.Tensor, specimen_idx: int, split: str = "train") -> torch.Tensor:
        if not self.should_augment(split):
            return coords
        
        generator = self._get_generator(specimen_idx)
        
        # Normalize z-axis
        if self.normalize_z_axis_enabled:
            coords = self.normalize_z_axis(coords)

        # Apply z-shift
        if self.z_shift_distribution == "beta":
            z_shift = self._sample_beta(self.z_shift_beta_alpha, self.z_shift_beta_beta, self.z_shift_range[0], self.z_shift_range[1], generator)
        else:
            z_shift = self._sample_uniform(self.z_shift_range[0], self.z_shift_range[1], generator)
        coords = self.apply_z_shift(coords, z_shift)
        
        # Apply uniform scaling
        if self.uniform_scale_distribution == "beta":
            uniform_scale = self._sample_beta(self.uniform_scale_beta_alpha, self.uniform_scale_beta_beta, self.uniform_scale_range[0], self.uniform_scale_range[1], generator)
        else:
            uniform_scale = self._sample_uniform(self.uniform_scale_range[0], self.uniform_scale_range[1], generator)
        coords = self.apply_uniform_scaling(coords, uniform_scale)
        
        # Apply z-rotation
        if self.rotation_distribution == "beta":
            angle_deg = self._sample_beta(self.rotation_beta_alpha, self.rotation_beta_beta, self.z_rotation_range[0], self.z_rotation_range[1], generator)
        else:
            angle_deg = self._sample_uniform(self.z_rotation_range[0], self.z_rotation_range[1], generator)
        angle_rad = np.deg2rad(angle_deg)
        coords = self.apply_z_rotation(coords, angle_rad)
        
        # Apply post-rotation xy-plane scaling with random orientation
        if self.post_rotation_xy_scale_distribution == "beta":
            xy_scale = self._sample_beta(self.post_rotation_xy_scale_beta_alpha, self.post_rotation_xy_scale_beta_beta, self.post_rotation_xy_scale_range[0], self.post_rotation_xy_scale_range[1], generator)
        else:
            xy_scale = self._sample_uniform(self.post_rotation_xy_scale_range[0], self.post_rotation_xy_scale_range[1], generator)
        orientation_deg = self._sample_uniform(self.xy_scale_orientation_range[0], self.xy_scale_orientation_range[1], generator)
        orientation_rad = np.deg2rad(orientation_deg)
        coords = self.apply_post_rotation_xy_scaling(coords, xy_scale, orientation_rad)

        # XY planar jitter (adds a small translation in the xy plane)
        if self.xy_jitter_mode != "none":
            coords = self.apply_xy_jitter(coords, generator)
        
        return coords