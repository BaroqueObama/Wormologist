import torch


class CoordinateEncoder:
    """Utility for transforming 3D coordinates into different representations."""
    
    def __init__(self, coordinate_system: str = "cylindrical", normalize: bool = False):
        self.coordinate_system = coordinate_system
        self.normalize = normalize
    
    
    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        """Transform raw 3D coordinates to the specified representation."""
        
        if self.coordinate_system == "cartesian":
            transformed = coords
        elif self.coordinate_system == "cylindrical":
            transformed = self.to_cylindrical(coords)
        else:
            raise ValueError(f"Unknown coordinate system: {self.coordinate_system}")
        
        if self.normalize:
            transformed = self.normalize_features(transformed)
        
        return transformed
    
    
    def to_cylindrical(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Convert Cartesian coordinates to cylindrical representation.
        
        Args:
            coords: Cartesian coordinates [N, 3] with columns [x, y, z]
            
        Returns:
            Cylindrical features [N, 4] with columns [sin(theta), cos(theta), radius, z]
        """
        
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        
        radius = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        cylindrical = torch.stack([sin_theta, cos_theta, radius, z], dim=1)
        
        return cylindrical
    
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to have zero mean and unit variance."""

        features = features - features.mean(dim=0, keepdim=True)
        
        std = features.std(dim=0, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        features = features / std
        
        return features
    
    
    @staticmethod
    def get_output_dim(coordinate_system: str) -> int:
        """Get the output dimension for a given coordinate system."""
        
        if coordinate_system == "cartesian":
            return 3
        elif coordinate_system == "cylindrical":
            return 4
        else:
            raise ValueError(f"Unknown coordinate system: {coordinate_system}")
