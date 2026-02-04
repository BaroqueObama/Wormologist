from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# Configs:
# - ModelConfig
# - TrainingConfig
# - OptimizerConfig
# - SchedulerConfig
# - SinkhornConfig
# - LossConfig
# - AugmentationConfig
# - CurriculumConfig
# - DataConfig
# - GraphConfig
# - SystemConfig
# - StorageConfig
# - ExperimentConfig
# - TaskConfig

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Core dimensions
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 8
    
    # MLA (Multi-head Latent Attention) configuration
    use_mla_layers: List[int] = field(default_factory=lambda: [4, 5, 6, 7])  # Which layers use MLA (e.g., [8,9,10,11,12,13,14,15])
    mla_kv_latent_dim: int = 64  # KV compression dimension (d_c in DeepSeek-V2)
    
    # Hybrid configuration for MLA + edge correction
    edge_correction_dim_ratio: float = 0.5  # Edge attention uses 1/2 of hidden_dim
    learnable_edge_weight: bool = True  # Whether edge weight is learnable
    initial_edge_weight: float = 0.5  # Initial value for edge correction weight
    
    # FFN parameters
    ffn_dim_multiplier: float = 4.0
    
    # Normalization strategy
    norm_style: str = "pre_post"  # Options: "post", "pre", "pre_post" (Gemma-style)
    
    # Positional encoding
    k_hops: int = 31  
    update_edge_repr: bool = True
    
    # Multi-scale RRWP configuration
    use_multiscale_pe: bool = False  # Enable multi-scale RRWP filtration
    num_scales: int = 4  # Number of different scales for filtration
    min_sigma: float = 0.1  # Minimum sigma for Gaussian kernel (very local)
    max_sigma: float = 0.7  # Maximum sigma for Gaussian kernel (regional)
    learnable_scales: bool = True  # Whether sigmas should be learnable parameters
    
    # Regularization
    attn_dropout: float = 0.0
    dropout: float = 0.0
    
    # Degree information (disabled by default, kept for potential future experiments)
    use_degree_scaler: bool = False
    
    # Output
    out_dim: int = 558
    cell_type_out_dim: int = 0  # Number of cell types (for cell_type task)
    
    def __post_init__(self):
        # Set default MLA layers if not specified (use MLA for later half of layers)
        if self.use_mla_layers is None:
            # For a 6-layer model, use MLA on layers 3,4,5
            if self.num_layers > 4:
                self.use_mla_layers = list(range(self.num_layers // 2, self.num_layers))
            else:
                self.use_mla_layers = []  # Don't use MLA for small models by default


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32  # Effective batch size (after gradient accumulation)
    micro_batch_size: int = 4  # Reduced from 4 to fit GPU memory
    epochs: int = 1
    gradient_clip_val: float = 1.0
    val_check_interval: int = 256  # Run validation every N training batches
    limit_val_batches: int = 64  # Only use 64 batches for validation (64 * micro_batch_size samples)
    checkpoint_interval: int = 256  # Save checkpoint every N training batches
    seed: int = 42
    
    def accumulation_steps(self, num_gpus: int = 1) -> int:
        """
        Calculate gradient accumulation steps from batch sizes and GPU count.
        micro_batch_size * accumulation_steps * num_gpus = batch_size
        """
        return max(1, self.batch_size // (self.micro_batch_size * num_gpus))


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer_type: str = "muon"  # "adamw", "muon"
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    momentum: float = 0.9  # For SGD/Muon
    
    # Muon-specific parameters
    muon_momentum: float = 0.95  # Momentum for Muon optimizer
    muon_nesterov: bool = True  # Use Nesterov momentum in Muon
    muon_ns_steps: int = 5  # Newton-Schulz iteration steps
    muon_lr: float = 0.003  # Learning rate specifically for Muon params
    
    # AdamW parameters when using Muon (for non-2D params and specified AdamW params)
    muon_adamw_lr: Optional[float] = None  # AdamW LR when using Muon (defaults to learning_rate)
    muon_adamw_betas: List[float] = field(default_factory=lambda: [0.9, 0.95])  # AdamW betas for Muon
    muon_adamw_eps: float = 1e-8  # AdamW epsilon for Muon
    muon_adamw_wd: float = 0.01  # AdamW weight decay for Muon


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    warmup_batches: int = 512  # Number of batches for warmup (was warmup_epochs)
    warmup_start_lr: float = 1e-6
    min_lr: float = 0.0005  # Increased from 1e-6 to maintain reasonable LR throughout training
    
    # Total training batches (will be calculated based on dataset size)
    # This is used for cosine annealing schedule
    total_training_batches: Optional[int] = 32768 # TODO: make more flexible to not hardcode


@dataclass
class SinkhornConfig:
    """Sinkhorn matcher configuration"""
    num_canonical: int = 558  # Number of canonical IDs to match to
    num_iterations: int = 5
    init_temperature: float = 1.0
    final_temperature: float = 0.01
    temperature_warmup_steps: int = 100
    dustbin_weight: float = 1.0
    epsilon: float = 1e-8
    one_to_one: bool = False  # If True, bias towards one-to-one matching; if False, allow many-to-one # TODO: fiddle with this


@dataclass
class LossConfig:
    """Loss function configuration"""
    class_weight: float = 1.0
    assignment_weight: float = 1.0
    label_smoothing: float = 0.1
    worm_averaged_loss: bool = True  # Average loss per worm (vs per node)
    use_superglue_loss: bool = False # supervise full Sinkhorn matrix (one-to-one required)
    # Flags for Sinkhorn finetuning
    use_topk_sinkhorn_mask: bool = False
    topk_sinkhorn_k: int = 5
    cell_type_loss_weight: float = 0.0  # Weight for auxiliary cell type classification loss

@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    enabled: bool = True  # Whether to apply augmentation

    normalize_z_axis: bool = True
    
    # Z-axis shift parameters (applied after normalizing min z to 0)
    z_shift_range: List[float] = field(default_factory=lambda: [0.005, 0.035]) # Initialize this with 0.02 and 0.020001
    z_shift_distribution: str = "beta"  # "uniform" or "beta"
    z_shift_beta_alpha: float = 1.0  # Beta distribution shape parameter
    z_shift_beta_beta: float = 1.0   # Beta distribution shape parameter
    
    # Uniform scaling parameters (applied to all axes before rotation)
    uniform_scale_range: List[float] = field(default_factory=lambda: [0.88, 1.05]) # 1 to 1
    uniform_scale_distribution: str = "beta"  # "uniform" or "beta"
    uniform_scale_beta_alpha: float = 1.0  # Beta distribution shape parameter
    uniform_scale_beta_beta: float = 1.0   # Beta distribution shape parameter
    
    # Rotation range for z-axis in degrees
    z_rotation_range: List[float] = field(default_factory=lambda: [-10.0, 10.0]) # 0 to 0
    rotation_distribution: str = "uniform"  # "uniform" or "beta" for rotation
    rotation_beta_alpha: float = 1.5  # Beta distribution shape parameter for rotation
    rotation_beta_beta: float = 1.5   # Beta distribution shape parameter for rotation
    
    # Post-rotation xy-plane scaling (scales along random orientation, orthogonal scales by 1/scale)
    post_rotation_xy_scale_range: List[float] = field(default_factory=lambda: [0.8, 1.2]) # 1 to 1
    post_rotation_xy_scale_distribution: str = "beta"  # "uniform" or "beta"
    post_rotation_xy_scale_beta_alpha: float = 1.2  # Beta distribution shape parameter
    post_rotation_xy_scale_beta_beta: float = 1.2   # Beta distribution shape parameter
    # Random orientation for xy scaling (in degrees, 0-360)
    xy_scale_orientation_range: List[float] = field(default_factory=lambda: [0.0, 360.0]) # 0 to 0

    # XY jitter (planar translation)
    xy_jitter_mode: str = "none"        # "none", "normal", or "uniform"
    xy_jitter_std: float = 0.0          # σ for normal mode
    xy_jitter_range: float = 0.0        # half-width for uniform mode
    
    # Control
    apply_to_splits: List[str] = field(default_factory=lambda: ["train", "val"])  # Only apply to these splits
    seed: int = 42  # Random seed for reproducibility


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning with node dropping."""
    # Enable/disable curriculum learning
    enabled: bool = True

    # New parameters for sliced subgraphs curriculum learning
    ######################
    subgraph_strategy: str = "random"          # "random" keeps existing behaviour; "sliced" enables PCA slicing
    slice_max_n_slices: int = 40               
    slice_thickness: Optional[float] = None    # None ⇒ auto-compute from augmented AP span / n_slices
    slice_shift: float = 0.0
    slice_crop_axis: Optional[str] = 'random'      # {"x","y","random",None}
    slice_crop_side: str = "positive"
    slice_crop_fraction: float = 0.0
    slice_seed: int = 42
    slice_profiles: List[Dict[str, Any]] = field(default_factory=lambda: [
    {"thickness_fraction": 0.007, "crop_fraction_range": [0.5, 1.0]},
    {"thickness_fraction": 0.008, "crop_fraction_range": [0.0, 1.0]},
    {"thickness_fraction": 0.009, "crop_fraction_range": [0.0, 0.5]},
    ])
    interpolate_crop_fraction: bool = False
    ######################
    
    # Visibility range
    start_visibility: float = 1.0  # 100% visible at start
    end_visibility: float = 0.15    # 15% visible at curriculum end
    
    # Batch-based progression
    warmup_batches: int = 128      # Full visibility for initial batches
    curriculum_batches: int = 512  # Batches to go from start to end visibility
    cooldown_batches: int = 32768   # Batches at end visibility distribution
    # After cooldown, remaining batches use uniform distribution
    
    # Distribution configuration for training
    train_distribution: str = "beta"  # "beta" or "truncated_normal"
    
    # Beta distribution parameters for curriculum phase
    curriculum_beta_alpha: float = 0.9  # Alpha parameter for curriculum phase
    curriculum_beta_beta: float = 1.6   # Beta parameter for curriculum phase
    
    # Beta distribution parameters for cooldown phase
    cooldown_beta_alpha: float = 1.0  # Alpha parameter for cooldown phase
    cooldown_beta_beta: float = 2.0   # Beta parameter for cooldown phase
    
    # Truncated normal parameters (alternative to beta)
    truncated_std: float = 0.1  # Standard deviation for truncated normal
    
    # Validation configuration  
    val_distribution: str = "uniform"  # Always uniform for validation
    val_min_visibility: float = 0.18    # Minimum visibility for validation
    val_max_visibility: float = 0.22    # Maximum visibility for validation
    
    # Random seed for reproducibility
    seed: int = 42
    
    def get_phase(self, global_batch: int) -> str:
        """Determine phase of curriculum."""
        
        if global_batch < self.warmup_batches:
            return "warmup"
        elif global_batch < self.warmup_batches + self.curriculum_batches:
            return "curriculum"
        elif global_batch < self.warmup_batches + self.curriculum_batches + self.cooldown_batches:
            return "cooldown"
        else:
            return "uniform"
    
    def get_curriculum_target(self, global_batch: int) -> float:
        """Calculate target visibility for current batch in training."""
        
        phase = self.get_phase(global_batch)
        
        if phase == "warmup":
            return self.start_visibility
        elif phase == "curriculum":
            progress_batches = global_batch - self.warmup_batches
            progress_ratio = progress_batches / self.curriculum_batches
            visibility = self.start_visibility - progress_ratio * (self.start_visibility - self.end_visibility)
            return max(self.end_visibility, min(self.start_visibility, visibility))
        elif phase == "cooldown":
            return self.end_visibility
        else:
            return None


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_path: str = "path/to/data"
    coordinate_system: str = "cylindrical"  # "cartesian" or "cylindrical"
    normalize_coords: bool = False  # Disabled to avoid issues with subgraphs
    use_cell_type_features: bool = True  # Include one-hot encoded cell type features
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class GraphConfig:
    """Graph construction configuration"""
    distance_kernel: str = "gaussian"  # "gaussian", "inverse", "linear"
    kernel_sigma: float = 0.5
    use_random_walks: bool = True
    walk_length: int = 31  # Should match pe_dim for RRWP # TODO: Fix to match k_hops


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    device: str = "cuda"
    mixed_precision: str = "bf16-mixed"  # "bf16", "fp16", "fp32"
    distributed: bool = True
    num_gpus: int = 4


@dataclass
class StorageConfig:
    """Storage and output configuration"""
    checkpoint_dir: str = "path/to/checkpoints"  # Base directory for checkpoints
    wandb_dir: str = "path/to/wandb"  # Base directory for wandb files
    use_experiment_subdir: bool = True  # Create subdirectory with experiment name
    save_config_copy: bool = True  # Save config.yaml with checkpoints
    
    def get_checkpoint_path(self, experiment_name: str) -> str:
        if self.use_experiment_subdir:
            return f"{self.checkpoint_dir}/{experiment_name}"
        return self.checkpoint_dir
    
    def get_wandb_path(self, experiment_name: str) -> str:
        if self.use_experiment_subdir:
            return f"{self.wandb_dir}/{experiment_name}"
        return self.wandb_dir


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    experiment_name: str = "baseline"
    wandb_project: str = "celegans"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["refactored", "sinkhorn"])
    log_interval: int = 10
    save_top_k: int = 3
    monitor_metric: str = "val/accuracy"
    monitor_mode: str = "max"

@dataclass
class TaskConfig:
    target: str = "canonical"      # "canonical" or "cell_type"
    num_classes: int = 7           # 0..6 cell types

@dataclass
class Config:
    """Master configuration containing all sub-configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    sinkhorn: SinkhornConfig = field(default_factory=SinkhornConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary"""
        config = cls()
        
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "optimizer" in config_dict:
            config.optimizer = OptimizerConfig(**config_dict["optimizer"])
        if "scheduler" in config_dict:
            config.scheduler = SchedulerConfig(**config_dict["scheduler"])
        if "sinkhorn" in config_dict:
            config.sinkhorn = SinkhornConfig(**config_dict["sinkhorn"])
        if "loss" in config_dict:
            config.loss = LossConfig(**config_dict["loss"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "augmentation" in config_dict:
            config.augmentation = AugmentationConfig(**config_dict["augmentation"])
        if "curriculum" in config_dict:
            config.curriculum = CurriculumConfig(**config_dict["curriculum"])
        if "graph" in config_dict:
            config.graph = GraphConfig(**config_dict["graph"])
        if "system" in config_dict:
            config.system = SystemConfig(**config_dict["system"])
        if "storage" in config_dict:
            config.storage = StorageConfig(**config_dict["storage"])
        if "experiment" in config_dict:
            config.experiment = ExperimentConfig(**config_dict["experiment"])
        if "task" in config_dict:  # ← add this block
            config.task = TaskConfig(**config_dict["task"])

        return config
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving/logging"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "sinkhorn": self.sinkhorn.__dict__,
            "loss": self.loss.__dict__,
            "data": self.data.__dict__,
            "augmentation": self.augmentation.__dict__,
            "curriculum": self.curriculum.__dict__,
            "graph": self.graph.__dict__,
            "system": self.system.__dict__,
            "storage": self.storage.__dict__,
            "experiment": self.experiment.__dict__,
            "task": self.task.__dict__,
        }
    
    def validate(self):
        """Validate configuration consistency"""
        # Model validations
        assert self.model.hidden_dim > 0, "Hidden dim must be positive"
        assert self.model.num_layers > 0, "Number of layers must be positive"
        assert self.model.num_heads > 0, "Number of heads must be positive"
        assert self.model.hidden_dim % self.model.num_heads == 0, "Hidden dim must be divisible by num_heads"
        
        # Model-Sinkhorn consistency check
        if self.task.target == "canonical":
            assert self.model.out_dim == self.sinkhorn.num_canonical, \
                f"Model out_dim ({self.model.out_dim}) must match Sinkhorn num_canonical ({self.sinkhorn.num_canonical})"
        else:
            assert self.task.num_classes > 0, "num_classes must be positive for cell_type task"
        
        # Training validations
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.epochs > 0, "Epochs must be positive"
        assert self.training.val_check_interval > 0, "Val check interval must be positive"
        assert self.training.limit_val_batches > 0, "Limit val batches must be positive"
        
        # Optimizer validations
        assert self.optimizer.learning_rate > 0, "Learning rate must be positive"
        assert self.optimizer.weight_decay >= 0, "Weight decay must be non-negative"
        
        # Sinkhorn validations
        assert self.sinkhorn.num_canonical > 0, "Number of canonical IDs must be positive"
        assert self.sinkhorn.num_iterations > 0, "Sinkhorn iterations must be positive"
        assert self.sinkhorn.init_temperature >= self.sinkhorn.final_temperature, "Init temp must be >= final temp"
        
        # Loss validations
        assert self.loss.class_weight >= 0, "Class weight must be non-negative"
        assert self.loss.assignment_weight >= 0, "Assignment weight must be non-negative"
        assert 0 <= self.loss.label_smoothing < 1, "Label smoothing must be in [0, 1)"
        
        # Data validations
        assert self.data.num_workers >= 0, "Number of workers must be non-negative"
        
        # System validations
        assert self.system.device in ["cuda", "cpu"], "Device must be 'cuda' or 'cpu'"
        assert self.system.mixed_precision in ["bf16", "fp16", "fp32", "bf16-mixed"], "Invalid mixed precision"
        
        return True
