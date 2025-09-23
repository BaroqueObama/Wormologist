import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback


class CurriculumCallback(Callback):
    """Callback to update dataset's global batch counter for curriculum learning."""
    
    def __init__(self, curriculum_config):
        super().__init__()
        self.curriculum_config = curriculum_config
        self.enabled = curriculum_config.enabled
    
    
    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: any, batch_idx: int) -> None:
        """Updates dataset's global batch counter for curriculum phase."""
        
        if not self.enabled:
            return
        
        global_batch = pl_module.global_batch_counter
        
        if hasattr(trainer, 'train_dataloader'):
            dataloader = trainer.train_dataloader
            if hasattr(dataloader, 'dataset'):
                dataloader.dataset.current_global_batch = global_batch
    
    def on_validation_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Ensures validation doesn't use global_batch (will trigger uniform sampling)."""
        
        if not self.enabled:
            return
        
        if hasattr(trainer, 'val_dataloaders'):
            dataloaders = trainer.val_dataloaders
            if dataloaders:
                dataloader = dataloaders[dataloader_idx] if isinstance(dataloaders, list) else dataloaders
                if hasattr(dataloader, 'dataset'):
                    dataloader.dataset.current_global_batch = None
