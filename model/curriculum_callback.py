import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback


class CurriculumCallback(Callback):
    """Callback to update curriculum visibility rate before each training batch for data loader."""
    
    def __init__(self, curriculum_config, node_dropper):
        super().__init__()
        self.curriculum_config = curriculum_config
        self.node_dropper = node_dropper
        self.enabled = curriculum_config.enabled
    
    
    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: any, batch_idx: int) -> None:
        """Updates dataset's visibility rate based on curriculum phase."""
        
        if not self.enabled:
            return
        
        global_batch = pl_module.global_batch_counter
        visibility_rate = self.node_dropper.get_visibility_rate(global_batch, "train")
        
        if hasattr(trainer, 'train_dataloader'):
            dataloader = trainer.train_dataloader
            if hasattr(dataloader, 'dataset'):
                dataloader.dataset.current_visibility_rate = visibility_rate
    
    # TODO: Check
    def on_validation_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Ensures validation uses uniform sampling (rate = None triggers uniform sampling)."""
        
        if not self.enabled:
            return
        
        if hasattr(trainer, 'val_dataloaders'):
            dataloaders = trainer.val_dataloaders
            if dataloaders:
                dataloader = dataloaders[dataloader_idx] if isinstance(dataloaders, list) else dataloaders
                if hasattr(dataloader, 'dataset'):
                    dataloader.dataset.current_visibility_rate = None
