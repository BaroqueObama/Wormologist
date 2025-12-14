import yaml
from pathlib import Path
from typing import Union, Dict, Any
from config.config import (
    Config, ModelConfig, TrainingConfig, OptimizerConfig, 
    SchedulerConfig, SinkhornConfig, LossConfig, DataConfig,
    AugmentationConfig, CurriculumConfig, GraphConfig, 
    SystemConfig, StorageConfig, ExperimentConfig, TaskConfig
)


def convert_yaml_value(value: Any) -> Any:
    if isinstance(value, str):
        if 'e' in value.lower() or 'E' in value:
            try:
                return float(value)
            except ValueError:
                pass
        elif value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
    elif isinstance(value, list):
        return [convert_yaml_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_yaml_value(v) for k, v in value.items()}
    
    return value


def load_yaml_config(path: Union[str, Path]) -> Config:
    """Load a configuration from a YAML file."""
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        yaml_dict = yaml.safe_load(f) or {}
    
    yaml_dict = convert_yaml_value(yaml_dict)
    
    section_classes = {
        'model': ModelConfig,
        'training': TrainingConfig,
        'optimizer': OptimizerConfig,
        'scheduler': SchedulerConfig,
        'sinkhorn': SinkhornConfig,
        'loss': LossConfig,
        'data': DataConfig,
        'augmentation': AugmentationConfig,
        'curriculum': CurriculumConfig,
        'graph': GraphConfig,
        'system': SystemConfig,
        'storage': StorageConfig,
        'experiment': ExperimentConfig,
        'task': TaskConfig,
    }
    
    config = Config()
    
    for section_name, section_dict in yaml_dict.items():
        if section_name in section_classes and isinstance(section_dict, dict):
            section_class = section_classes[section_name]
            current_section = getattr(config, section_name)
            merged_dict = {**current_section.__dict__, **section_dict}
            new_section = section_class(**merged_dict)
            setattr(config, section_name, new_section)
    
    return config


def save_yaml_config(config: Config, path: Union[str, Path], include_defaults: bool = False):
    """Save a configuration to a YAML file."""
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if include_defaults:
        config_dict = config.to_dict()
    else:
        # Only save values that differ from defaults
        default_config = Config()
        default_dict = default_config.to_dict()
        config_dict = config.to_dict()
        
        # Remove values that match defaults
        filtered_dict = {}
        for section_name, section_dict in config_dict.items():
            if section_name in default_dict:
                filtered_section = {}
                for key, value in section_dict.items():
                    if key in default_dict[section_name]:
                        if value != default_dict[section_name][key]:
                            filtered_section[key] = value
                    else:
                        filtered_section[key] = value
                if filtered_section:
                    filtered_dict[section_name] = filtered_section
            else:
                filtered_dict[section_name] = section_dict
        config_dict = filtered_dict
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)


def apply_cli_overrides(config: Config, overrides: Dict[str, str]):
    """Apply command-line overrides to a configuration."""
    
    for override_key, override_value in overrides.items():
        keys = override_key.split('.')
        
        if len(keys) == 2:
            section_name, param_name = keys
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                if hasattr(section, param_name):
                    # Parse the value to appropriate type
                    parsed_value = parse_value(override_value)
                    setattr(section, param_name, parsed_value)
                else:
                    print(f"Warning: Parameter '{param_name}' not found in section '{section_name}'")
            else:
                print(f"Warning: Section '{section_name}' not found in config")
        else:
            print(f"Warning: Invalid override format '{override_key}'. Expected 'section.param'")


def parse_value(value: str) -> Any:
    """Parse a string value to its appropriate type."""

    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    
    try:
        import ast
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value