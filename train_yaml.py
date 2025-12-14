#!/usr/bin/env python

import argparse
from pathlib import Path
import yaml

from config.config import Config
from config.yaml_config import load_yaml_config, save_yaml_config, apply_cli_overrides
from loader.data_loader import create_data_loader
from model.lightning_model import train


def main():
    parser = argparse.ArgumentParser(
        description='Train C. elegans model with YAML configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with a YAML configuration
  python train_yaml.py --config configs/baseline.yaml
  
  # Override specific parameters
  python train_yaml.py --config configs/baseline.yaml --set model.hidden_dim=256
  
  # Use default config with overrides
  python train_yaml.py --set experiment.experiment_name=test --set training.epochs=2
  
  # Save current configuration
  python train_yaml.py --config configs/baseline.yaml --save_config my_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--set', action='append', dest='overrides',
                       help='Override config values (format: section.param=value)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_config', type=str, default=None,
                       help='Save the final configuration to a file and exit')
    parser.add_argument('--print_config', action='store_true',
                       help='Print the final configuration and exit')
    
    args = parser.parse_args()
    
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_yaml_config(args.config)
    else:
        print("Using default configuration")
        config = Config()
    
    
    # Apply command-line overrides
    if args.overrides:
        overrides = {}
        for override in args.overrides:
            if '=' not in override:
                print(f"Warning: Invalid override format '{override}'. Expected 'section.param=value'")
                continue
            key, value = override.split('=', 1)
            overrides[key] = value
        
        print(f"Applying overrides: {list(overrides.keys())}")
        apply_cli_overrides(config, overrides)
    
    
    # Validate configuration
    config.validate()
    
    
    # Handle utility options
    if args.print_config:
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False))
        return
    
    if args.save_config:
        save_yaml_config(config, args.save_config, include_defaults=False)
        print(f"Configuration saved to: {args.save_config}")
        return
    
    
    # Print experiment info
    print("\n" + "="*60)
    print("C. ELEGANS NUCLEI ASSIGNMENT TRAINING")
    print("="*60)
    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Model: {config.model.num_layers}L-{config.model.hidden_dim}D-{config.model.num_heads}H")
    print(f"Batch: {config.training.batch_size} (micro: {config.training.micro_batch_size})")
    print(f"Epochs: {config.training.epochs}")
    print(f"Optimizer: {config.optimizer.optimizer_type}")
    print(f"Learning rate: {config.optimizer.learning_rate}")
    if config.optimizer.optimizer_type == "muon":
        print(f"Muon LR: {config.optimizer.muon_lr}")
    print(f"Curriculum: {'Enabled' if config.curriculum.enabled else 'Disabled'}")
    print(f"Augmentation: {'Enabled' if config.augmentation.enabled else 'Disabled'}")
    print("="*60 + "\n")
    
    
    # Save config for reproducibility
    checkpoint_dir = Path(config.storage.get_checkpoint_path(config.experiment.experiment_name))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if config.storage.save_config_copy:
        save_yaml_config(config, checkpoint_dir / "config.yaml", include_defaults=True)
        print(f"Full configuration saved to: {checkpoint_dir}/config.yaml\n")
    
    
    # Create data loaders
    print("Creating data loaders:")
    
    train_loader = create_data_loader(
        data_path=config.data.data_path,
        split='train',
        batch_size=config.training.micro_batch_size,
        num_workers=config.data.num_workers,
        distributed=(config.system.num_gpus > 1),
        normalize_coords=config.data.normalize_coords,
        augmentation_config=config.augmentation if config.augmentation.enabled else None,
        curriculum_config=config.curriculum if config.curriculum.enabled else None,
        verbose=True,
        target=config.task.target,
        use_cell_type_features=config.data.use_cell_type_features
    )
    
    val_loader = create_data_loader(
        data_path=config.data.data_path,
        split='val',
        batch_size=config.training.micro_batch_size,
        num_workers=config.data.num_workers,
        distributed=False,
        normalize_coords=config.data.normalize_coords,
        augmentation_config=config.augmentation if config.augmentation.enabled else None,
        curriculum_config=config.curriculum if config.curriculum.enabled else None,
        verbose=False,
        target=config.task.target,
        use_cell_type_features=config.data.use_cell_type_features
    )
    
    
    # TODO: Fix estimate as IterableDataset doesn't have exact length
    print(f"Train loader: Processing sharded data from {config.data.data_path}/train/")
    print(f"Val loader: Processing sharded data from {config.data.data_path}/val/")
    
    
    # Train
    print("\nStarting training:")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        model, trainer = train(config, train_loader, val_loader, resume_from=args.resume_from)
    else:
        model, trainer = train(config, train_loader, val_loader)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    
    
    # Test if available
    try:
        test_loader = create_data_loader(
            data_path=config.data.data_path,
            split='test',
            batch_size=config.training.micro_batch_size,
            num_workers=config.data.num_workers,
            distributed=False,
            normalize_coords=config.data.normalize_coords,
            augmentation_config=config.augmentation if config.augmentation.enabled else None,
            verbose=False,
            target=config.task.target,
            use_cell_type_features=config.data.use_cell_type_features
        )
        print("\nRunning test evaluation:")
        trainer.test(model, test_loader)
    except ValueError:
        print("\nNo test set found.")
    
    print("\nComplete")


if __name__ == '__main__':
    main()
