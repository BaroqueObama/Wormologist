import torch
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any
import math

from loader.graph_encoder import GraphEncoder
from loader.coordinate_encoder import CoordinateEncoder
from loader.cell_types import NUM_CELL_TYPES, CELL_TYPES_FINE
from model.model import Wormologist
from model.pe import PEModule
from model.multiscale_pe import MultiScaleRRWPFiltration
from model.sinkhorn import AssignmentLoss, compute_assignment_accuracy
from model.curriculum_callback import CurriculumCallback
from config.config import Config
from optimizers.muon import Muon
import copy
import torch.nn as nn

class LightningModel(L.LightningModule):
    """Lightning module for training."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Track global batch counter for curriculum learning
        self.global_batch_counter = 0
        
        # Initialize PE modules based on config
        if config.model.use_multiscale_pe:
            self.pe_module = MultiScaleRRWPFiltration(
                num_scales=config.model.num_scales,
                num_steps=config.model.k_hops,
                min_sigma=config.model.min_sigma,
                max_sigma=config.model.max_sigma,
                learnable_scales=config.model.learnable_scales
            )
            pe_node_dim = config.model.num_scales * config.model.k_hops
            pe_edge_dim = config.model.num_scales * config.model.k_hops
        else:
            self.pe_module = PEModule(k_hops=config.model.k_hops)
            pe_node_dim = config.model.k_hops
            pe_edge_dim = config.model.k_hops
        
        base_input_dim = CoordinateEncoder.get_output_dim(config.data.coordinate_system)
        
        # Add cell type feature dimensions if enabled
        if config.data.use_cell_type_features:
            base_input_dim += NUM_CELL_TYPES  # One-hot encoded cell types
        
        node_input_dim = base_input_dim + pe_node_dim
        
        edge_input_dim = 4 + pe_edge_dim # Edge features: 4D (distance + direction vector) + PE features
        
        if config.task.target == "cell_type":
            config.model.out_dim = config.task.num_classes

        self.model = Wormologist(
            config=config.model,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim
        )

        self.cell_type_loss_weight = config.loss.cell_type_loss_weight
        self.ce_loss_celltype = nn.CrossEntropyLoss(label_smoothing=config.loss.label_smoothing)
        
        if config.task.target == "canonical":
            # Loss computation
            self.loss_fn = AssignmentLoss(
                num_canonical = config.sinkhorn.num_canonical,
                num_iterations = config.sinkhorn.num_iterations,
                temperature = config.sinkhorn.init_temperature,
                class_weight = config.loss.class_weight,
                assignment_weight = config.loss.assignment_weight,
                label_smoothing = config.loss.label_smoothing,
                worm_averaged_loss = config.loss.worm_averaged_loss,
                one_to_one = config.sinkhorn.one_to_one,
                use_superglue_loss = config.loss.use_superglue_loss,
                use_topk_sinkhorn_mask = config.loss.use_topk_sinkhorn_mask,
                topk_sinkhorn_k = config.loss.topk_sinkhorn_k,
            )
            
            # Share the matcher from loss_fn for inference to avoid unused parameters
            self.matcher = self.loss_fn.matcher
        else:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=config.loss.label_smoothing)
            self.matcher = None

    # TODO: This should be elswhere
    def _ce_loss_and_metrics(self, logits, labels, visible_mask):
        flat_mask = visible_mask.bool()
        flat_logits = logits[flat_mask]
        flat_labels = labels[flat_mask]
        loss = self.ce_loss(flat_logits, flat_labels)
        probs = torch.softmax(logits, dim=-1)
        vis_probs = probs[flat_mask]
        top1 = vis_probs.argmax(dim=-1)
        topk = torch.topk(vis_probs, k=min(5, vis_probs.shape[-1]), dim=-1).indices
        metrics = {
            'loss': loss,
            'accuracy': (top1 == flat_labels).float().mean(),
            'top3': (flat_labels.unsqueeze(1) == topk[:, :3]).any(dim=1).float().mean(),
            'top5': (flat_labels.unsqueeze(1) == topk).any(dim=1).float().mean(),
        }
        return loss, metrics

    def forward(self, batch: Dict) -> Dict:
        """Forward pass accepting batch dict from data loader or inference."""
        coords, edge_index, edge_attr, batch_assignment, _ = self.preprocess_batch(batch)
        return self.model(coords, edge_index, edge_attr, batch_assignment)
    
    
    def preprocess_batch(self, batch_data: Dict) -> tuple:
        """Prepare batch data for model processing, including RRWP computation."""
        
        pyg_batch = batch_data['batch']
        coords = pyg_batch.x  # Transformed coordinates (cylindrical or cartesian)
        batch_assignment = pyg_batch.batch
        labels = pyg_batch.y
        num_nodes = coords.size(0)
        
        # Data always provided by the data loader
        if not hasattr(pyg_batch, 'raw_coords'):
            raise RuntimeError("Missing 'raw_coords' attribute in batch data")
        
        raw_coords = pyg_batch.raw_coords
        
        if raw_coords.shape[1] != 3:
            raise ValueError("raw_coords must be 3D Cartesian coordinates")
        
        # Create fully connected graph with edge attributes using raw Cartesian coordinates
        edge_index, edge_attr = GraphEncoder.create_edges_with_attributes(raw_coords, batch_assignment)
        
        # Compute RRWP features
        if self.config.model.use_multiscale_pe:
            node_pe, edge_pe = self.pe_module.compute_multiscale_rrwp(raw_coords, edge_index)
        else:
            if edge_attr is not None and edge_attr.shape[0] > 0:
                distances = edge_attr[:, 0]  # First column is distance
                edge_weights = GraphEncoder.distance_to_weight(
                    distances, 
                    self.config.graph.distance_kernel,
                    self.config.graph.kernel_sigma
                )
            else:
                edge_weights = None
            
            node_pe, edge_pe = self.pe_module(edge_index, num_nodes, edge_weights)
        
        # Concatenate features
        coords_with_pe = torch.cat([coords, node_pe], dim=-1)
        
        if edge_attr is not None and edge_pe is not None:
            edge_attr_with_pe = torch.cat([edge_attr, edge_pe], dim=-1)
        else:
            edge_attr_with_pe = edge_pe
        
        return coords_with_pe, edge_index, edge_attr_with_pe, batch_assignment, labels
    
    
    def prepare_targets(self, labels: torch.Tensor, batch_assignment: torch.Tensor, visible_mask: torch.Tensor) -> Dict:
        """Prepare target dictionary for loss computation."""
        
        batch_size = batch_assignment.max().item() + 1
        max_nodes = visible_mask.shape[1]
        
        targets = {
            'labels': torch.zeros(batch_size, max_nodes, dtype=torch.long, device=self.device),
            'visible_mask': visible_mask
        }
        
        # Fill in labels for each graph
        for b in range(batch_size):
            mask = (batch_assignment == b)
            num_nodes_in_graph = mask.sum().item()
            if num_nodes_in_graph > 0:
                targets['labels'][b, :num_nodes_in_graph] = labels[mask]
        
        return targets
    
    
    def _process_batch(self, batch: Dict, compute_hungarian: bool = False):
        
        if self.config.task.target == "canonical":
            outputs = self(batch)

            pyg_batch = batch['batch']
            labels = pyg_batch.y
            batch_assignment = pyg_batch.batch
            visible_mask = batch['visible_mask']
            
            targets = self.prepare_targets(labels, batch_assignment, visible_mask)
            loss_dict = self.loss_fn(outputs, targets)
            metrics = compute_assignment_accuracy(outputs, targets, loss_dict['soft_assignments'], dustbin_weights=loss_dict.get('dustbin_weights', None), dustbin_row_weights=loss_dict.get('dustbin_row_weights', None), compute_hungarian=compute_hungarian, full_assignments=loss_dict.get('full_assignments', None))

            if self.cell_type_loss_weight > 0:
                if "cell_type_logits" not in outputs:
                    raise RuntimeError("cell_type_logits missing from model outputs")
                ct_logits = outputs["cell_type_logits"]
                visible_mask = targets['visible_mask'].bool()
                # map canonical labels -> cell-type labels
                labels_cpu = targets['labels'].detach().cpu().numpy()
                ct_np = CELL_TYPES_FINE[labels_cpu]
                ct_labels = torch.as_tensor(ct_np, device=ct_logits.device, dtype=torch.long)
                flat_logits = ct_logits[visible_mask]
                flat_labels = ct_labels[visible_mask]
                ct_loss = self.ce_loss_celltype(flat_logits, flat_labels)
                with torch.no_grad():
                    preds = flat_logits.argmax(dim=-1)
                    ct_acc = (preds == flat_labels).float().mean()
                loss_dict['cell_type_loss'] = ct_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + self.cell_type_loss_weight * ct_loss
                metrics['cell_type_accuracy'] = ct_acc

            return loss_dict, metrics

        else:
            outputs = self(batch)
            pyg_batch = batch['batch']
            labels = pyg_batch.y
            batch_assignment = pyg_batch.batch
            visible_mask = batch['visible_mask']
            targets = self.prepare_targets(labels, batch_assignment, visible_mask)  # pads to [B, max_nodes]
            loss, metrics = self._ce_loss_and_metrics(outputs['logits'], targets['labels'], visible_mask)
            return {'total_loss': loss}, metrics

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
    
        if self.config.task.target == "canonical":
            loss_dict, metrics = self._process_batch(batch)
        
            self._update_temperature()
            
            batch_size = len(batch['visible_mask'])
            
            # Log curriculum metrics if curriculum learning is enabled
            if self.config.curriculum.enabled and hasattr(batch['batch'], 'visibility_rate'):
                self.log('curriculum/avg_nodes', metrics['avg_nodes'], on_step=True, batch_size=batch_size)

                pyg_batch = batch['batch']
                # Extract visibility rates from the batch
                visibility_rates = []
                for i in range(len(batch['visible_mask'])):
                    # Get nodes for this sample in the batch
                    mask = (pyg_batch.batch == i)
                    if mask.any():
                        idx = mask.nonzero()[0].item()
                        if hasattr(pyg_batch, 'visibility_rate') and idx < len(pyg_batch.visibility_rate):
                            visibility_rates.append(pyg_batch.visibility_rate[idx].item())
                
                if visibility_rates:
                    avg_visibility = sum(visibility_rates) / len(visibility_rates)
                    self.log('curriculum/avg_visibility', avg_visibility, on_step=True, batch_size=batch_size)
                    self.log('curriculum/min_visibility', min(visibility_rates), on_step=True, batch_size=batch_size)
                    self.log('curriculum/max_visibility', max(visibility_rates), on_step=True, batch_size=batch_size)
                    
                    phase = self.config.curriculum.get_phase(self.global_batch_counter)
                    target = self.config.curriculum.get_curriculum_target(self.global_batch_counter)
                    
                    phase_map = {'warmup': 0, 'curriculum': 1, 'cooldown': 2, 'uniform': 3} # Map phase to numeric for logging
                    self.log('curriculum/phase', phase_map.get(phase, -1), on_step=True, batch_size=batch_size)
                    if target is not None:
                        self.log('curriculum/target_visibility', target, on_step=True, batch_size=batch_size)
                    self.log('curriculum/global_batch', self.global_batch_counter, on_step=True, batch_size=batch_size)
            
            # Only increment global batch counter after gradient accumulation is complete
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                self.global_batch_counter += 1

            self.log('train/loss', loss_dict['total_loss'], on_step=True, on_epoch=True, batch_size=batch_size)

            if 'superglue_loss' in loss_dict:
                self.log('train/superglue_loss', loss_dict['superglue_loss'], batch_size=batch_size)
            else:
                self.log('train/class_loss', loss_dict['class_loss'], batch_size=batch_size)
                self.log('train/assignment_loss', loss_dict['assignment_loss'], batch_size=batch_size)
            
            self.log('train/accuracy', metrics['exact_accuracy'], prog_bar=True, batch_size=batch_size)
            self.log('train/top3_accuracy', metrics['top3_accuracy'], batch_size=batch_size)
            self.log('train/top5_accuracy', metrics['top5_accuracy'], batch_size=batch_size)
            self.log('train/temperature', self.loss_fn.matcher.temperature, batch_size=batch_size)

            self.log('train/accuracy_logits', metrics['exact_accuracy_logits'], batch_size=batch_size, sync_dist=True)
            self.log('train/top3_accuracy_logits', metrics['top3_accuracy_logits'], batch_size=batch_size, sync_dist=True)
            self.log('train/top5_accuracy_logits', metrics['top5_accuracy_logits'], batch_size=batch_size, sync_dist=True)

            # SuperGlue metrics (training)
            if 'sg_exact_match_acc' in metrics:
                self.log('train/sg_exact_match_acc', metrics['sg_exact_match_acc'], batch_size=batch_size)
                self.log('train/sg_num_matched', metrics['sg_num_matched'], batch_size=batch_size)
                self.log('train/sg_num_unmatched_rows', metrics['sg_num_unmatched_rows'], batch_size=batch_size)
                self.log('train/sg_num_unmatched_cols', metrics['sg_num_unmatched_cols'], batch_size=batch_size)

                self.log('train/sg_conf_matched', metrics['sg_conf_matched'], batch_size=batch_size)
                self.log('train/sg_conf_unmatched_rows', metrics['sg_conf_unmatched_rows'], batch_size=batch_size)
                self.log('train/sg_conf_unmatched_cols', metrics['sg_conf_unmatched_cols'], batch_size=batch_size)

                self.log('train/sg_loss_matched', metrics['sg_loss_matched'], batch_size=batch_size)
                self.log('train/sg_loss_unmatched_rows', metrics['sg_loss_unmatched_rows'], batch_size=batch_size)
                self.log('train/sg_loss_unmatched_cols', metrics['sg_loss_unmatched_cols'], batch_size=batch_size)

            # SuperGlue dustbin parameters (training)
            self.log('train/sg_dustbin_col_score', self.loss_fn.matcher.dustbin_col_score, batch_size=batch_size)
            if self.loss_fn.matcher.one_to_one:
                self.log('train/sg_dustbin_row_score', self.loss_fn.matcher.dustbin_row_score, batch_size=batch_size)
                self.log('train/sg_dustbin_corner_score', self.loss_fn.matcher.dustbin_corner_score, batch_size=batch_size)

            
            # Log multi-scale rrwp sigma values
            if self.config.model.use_multiscale_pe and hasattr(self.pe_module, 'log_sigmas'):
                sigmas = torch.exp(self.pe_module.log_sigmas).detach()
                for i, sigma in enumerate(sigmas):
                    self.log(f'multiscale/sigma_{i}', sigma.item(), on_step=True, batch_size=batch_size)
                self.log('multiscale/sigma_min', sigmas.min().item(), on_step=True, batch_size=batch_size)
                self.log('multiscale/sigma_max', sigmas.max().item(), on_step=True, batch_size=batch_size)
                self.log('multiscale/sigma_range', (sigmas.max() - sigmas.min()).item(), on_step=True, batch_size=batch_size)
            
            if 'cell_type_loss' in loss_dict:
                self.log('train/cell_type_loss', loss_dict['cell_type_loss'], on_step=True, batch_size=batch_size)
                self.log('train/cell_type_accuracy', metrics['cell_type_accuracy'], on_step=True, batch_size=batch_size)

            return loss_dict['total_loss']
        else:
            loss_dict, metrics = self._process_batch(batch)
            batch_size = len(batch['visible_mask'])
            # if self.config.curriculum.enabled and hasattr(batch['batch'], 'visibility_rate'):
            #     self.log('curriculum/avg_nodes', metrics.get('avg_nodes', 0), on_step=True, batch_size=batch_size)
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                self.global_batch_counter += 1
            self.log('train/loss', loss_dict['total_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
            self.log('train/accuracy', metrics['accuracy'], prog_bar=True, batch_size=batch_size)
            self.log('train/top3_accuracy', metrics['top3'], batch_size=batch_size)
            self.log('train/top5_accuracy', metrics['top5'], batch_size=batch_size)
            return loss_dict['total_loss']

           
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        if self.config.task.target == "canonical":
            loss_dict, metrics = self._process_batch(batch)
        
            batch_size = len(batch['visible_mask'])
            
            # Use sync_dist=True for proper aggregation across GPUs
            # Not sure is I should log all of these for the validation as well. Am I slowing it down?
            self.log('val/loss', loss_dict['total_loss'], on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log('val/accuracy', metrics['exact_accuracy'], on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top3_accuracy', metrics['top3_accuracy'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top5_accuracy', metrics['top5_accuracy'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top1_mass', metrics['node_avg_top1_mass'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top2_mass', metrics['node_avg_top2_mass'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top3_mass', metrics['node_avg_top3_mass'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top5_mass', metrics['node_avg_top5_mass'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top10_mass', metrics['node_avg_top10_mass'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top1_gap', metrics['node_avg_gap_top1_correct'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top1_rank', metrics['node_avg_rank_top1_correct'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top1_mass_sinkhorn', metrics['node_avg_top1_mass_sinkhorn'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top2_mass_sinkhorn', metrics['node_avg_top2_mass_sinkhorn'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top3_mass_sinkhorn', metrics['node_avg_top3_mass_sinkhorn'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top5_mass_sinkhorn', metrics['node_avg_top5_mass_sinkhorn'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top10_mass_sinkhorn', metrics['node_avg_top10_mass_sinkhorn'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top1_gap_sinkhorn', metrics['node_avg_gap_top1_correct_sinkhorn'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top1_rank_sinkhorn', metrics['node_avg_rank_top1_correct_sinkhorn'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/node_avg_dustbin_share', metrics['node_avg_dustbin_share'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            if 'canonical_avg_dustbin_row_share' in metrics:
                self.log('val/canonical_avg_dustbin_row_share', metrics['canonical_avg_dustbin_row_share'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/avg_nodes', metrics['avg_nodes'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            
            # SuperGlue metrics (validation)
            if 'sg_exact_match_acc' in metrics:
                self.log('val/sg_exact_match_acc', metrics['sg_exact_match_acc'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_num_matched', metrics['sg_num_matched'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_num_unmatched_rows', metrics['sg_num_unmatched_rows'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_num_unmatched_cols', metrics['sg_num_unmatched_cols'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)

                self.log('val/sg_conf_matched', metrics['sg_conf_matched'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_conf_unmatched_rows', metrics['sg_conf_unmatched_rows'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_conf_unmatched_cols', metrics['sg_conf_unmatched_cols'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)

                self.log('val/sg_loss_matched', metrics['sg_loss_matched'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_loss_unmatched_rows', metrics['sg_loss_unmatched_rows'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_loss_unmatched_cols', metrics['sg_loss_unmatched_cols'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)

            # SuperGlue dustbin parameters (validation)
            self.log('val/sg_dustbin_col_score', self.loss_fn.matcher.dustbin_col_score,
                    on_epoch=True, batch_size=batch_size, sync_dist=True)
            if self.loss_fn.matcher.one_to_one:
                self.log('val/sg_dustbin_row_score', self.loss_fn.matcher.dustbin_row_score,
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/sg_dustbin_corner_score', self.loss_fn.matcher.dustbin_corner_score,
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                
            self.log('val/accuracy_logits', metrics['exact_accuracy_logits'], batch_size=batch_size, sync_dist=True)
            self.log('val/top3_accuracy_logits', metrics['top3_accuracy_logits'], batch_size=batch_size, sync_dist=True)
            self.log('val/top5_accuracy_logits', metrics['top5_accuracy_logits'], batch_size=batch_size, sync_dist=True)

            if 'cell_type_loss' in loss_dict:
                self.log('val/cell_type_loss', loss_dict['cell_type_loss'], on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('val/cell_type_accuracy', metrics['cell_type_accuracy'], on_epoch=True, batch_size=batch_size, sync_dist=True)

        else:
            loss_dict, metrics = self._process_batch(batch)
            batch_size = len(batch['visible_mask'])
            self.log('val/loss', loss_dict['total_loss'], on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log('val/accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top3_accuracy', metrics['top3'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log('val/top5_accuracy', metrics['top5'], on_epoch=True, batch_size=batch_size, sync_dist=True)
            
    def test_step(self, batch: Dict, batch_idx: int) -> None:
        if self.config.task.target == "canonical":
            loss_dict, metrics = self._process_batch(batch, compute_hungarian=True)
        
            # Get actual batch size (number of graphs, not nodes)
            batch_size = len(batch['visible_mask'])
            
            # Use sync_dist=True for proper aggregation across GPUs in distributed setting
            self.log('test/loss', loss_dict['total_loss'], batch_size=batch_size, sync_dist=True)
            self.log('test/accuracy', metrics['exact_accuracy'], batch_size=batch_size, sync_dist=True)
            self.log('test/top3_accuracy', metrics['top3_accuracy'], batch_size=batch_size, sync_dist=True)
            self.log('test/top5_accuracy', metrics['top5_accuracy'], batch_size=batch_size, sync_dist=True)
            self.log('test/top1_mass', metrics['node_avg_top1_mass'], batch_size=batch_size, sync_dist=True)
            self.log('test/top2_mass', metrics['node_avg_top2_mass'], batch_size=batch_size, sync_dist=True)
            self.log('test/top3_mass', metrics['node_avg_top3_mass'], batch_size=batch_size, sync_dist=True)
            self.log('test/top5_mass', metrics['node_avg_top5_mass'], batch_size=batch_size, sync_dist=True)
            self.log('test/top10_mass', metrics['node_avg_top10_mass'], batch_size=batch_size, sync_dist=True)
            self.log('test/top1_gap', metrics['node_avg_gap_top1_correct'], batch_size=batch_size, sync_dist=True)
            self.log('test/top1_rank', metrics['node_avg_rank_top1_correct'], batch_size=batch_size, sync_dist=True)
            self.log('test/top1_mass_sinkhorn', metrics['node_avg_top1_mass_sinkhorn'], batch_size=batch_size, sync_dist=True)
            self.log('test/top2_mass_sinkhorn', metrics['node_avg_top2_mass_sinkhorn'], batch_size=batch_size, sync_dist=True)
            self.log('test/top3_mass_sinkhorn', metrics['node_avg_top3_mass_sinkhorn'], batch_size=batch_size, sync_dist=True)
            self.log('test/top5_mass_sinkhorn', metrics['node_avg_top5_mass_sinkhorn'], batch_size=batch_size, sync_dist=True)
            self.log('test/top10_mass_sinkhorn', metrics['node_avg_top10_mass_sinkhorn'], batch_size=batch_size, sync_dist=True)
            self.log('test/top1_gap_sinkhorn', metrics['node_avg_gap_top1_correct_sinkhorn'], batch_size=batch_size, sync_dist=True)
            self.log('test/top1_rank_sinkhorn', metrics['node_avg_rank_top1_correct_sinkhorn'], batch_size=batch_size, sync_dist=True)
            self.log('test/node_avg_dustbin_share', metrics['node_avg_dustbin_share'], batch_size=batch_size, sync_dist=True)
            self.log('test/avg_nodes', metrics['avg_nodes'], batch_size=batch_size, sync_dist=True)
            self.log("test/exact_accuracy_sum", metrics["exact_accuracy_sum"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/exact_accuracy_sum_sq", metrics["exact_accuracy_sum_sq"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/exact_accuracy_count", metrics["exact_accuracy_count"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/top3_accuracy_sum", metrics["top3_accuracy_sum"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/top3_accuracy_sum_sq", metrics["top3_accuracy_sum_sq"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/top3_accuracy_count", metrics["top3_accuracy_count"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/top5_accuracy_sum", metrics["top5_accuracy_sum"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/top5_accuracy_sum_sq", metrics["top5_accuracy_sum_sq"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            self.log("test/top5_accuracy_count", metrics["top5_accuracy_count"], batch_size=batch_size, reduce_fx=torch.sum, sync_dist=True)
            if 'canonical_avg_dustbin_row_share' in metrics:
                self.log('test/canonical_avg_dustbin_row_share', metrics['canonical_avg_dustbin_row_share'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
            if 'exact_accuracy_hungarian' in metrics:
                self.log('test/accuracy_hungarian', metrics['exact_accuracy_hungarian'], batch_size=batch_size, sync_dist=True)
            self.log('test/accuracy_logits', metrics['exact_accuracy_logits'], batch_size=batch_size, sync_dist=True)
            self.log('test/top3_accuracy_logits', metrics['top3_accuracy_logits'], batch_size=batch_size, sync_dist=True)
            self.log('test/top5_accuracy_logits', metrics['top5_accuracy_logits'], batch_size=batch_size, sync_dist=True)

            # SuperGlue metrics (test)
            if 'sg_exact_match_acc' in metrics:
                self.log('test/sg_exact_match_acc', metrics['sg_exact_match_acc'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_num_matched', metrics['sg_num_matched'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_num_unmatched_rows', metrics['sg_num_unmatched_rows'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_num_unmatched_cols', metrics['sg_num_unmatched_cols'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)

                self.log('test/sg_conf_matched', metrics['sg_conf_matched'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_conf_unmatched_rows', metrics['sg_conf_unmatched_rows'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_conf_unmatched_cols', metrics['sg_conf_unmatched_cols'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)

                self.log('test/sg_loss_matched', metrics['sg_loss_matched'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_loss_unmatched_rows', metrics['sg_loss_unmatched_rows'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_loss_unmatched_cols', metrics['sg_loss_unmatched_cols'],
                        on_epoch=True, batch_size=batch_size, sync_dist=True)

            # SuperGlue dustbin parameters (test)
            self.log('test/sg_dustbin_col_score', self.loss_fn.matcher.dustbin_col_score,
                    on_epoch=True, batch_size=batch_size, sync_dist=True)
            if self.loss_fn.matcher.one_to_one:
                self.log('test/sg_dustbin_row_score', self.loss_fn.matcher.dustbin_row_score,
                        on_epoch=True, batch_size=batch_size, sync_dist=True)
                self.log('test/sg_dustbin_corner_score', self.loss_fn.matcher.dustbin_corner_score,
                       on_epoch=True, batch_size=batch_size, sync_dist=True)
        
            if 'cell_type_loss' in loss_dict:
                self.log('test/cell_type_loss', loss_dict['cell_type_loss'], batch_size=batch_size, sync_dist=True)
                self.log('test/cell_type_accuracy', metrics['cell_type_accuracy'], batch_size=batch_size, sync_dist=True)

        else:
            loss_dict, metrics = self._process_batch(batch)
            batch_size = len(batch['visible_mask'])
            self.log('test/loss', loss_dict['total_loss'], batch_size=batch_size, sync_dist=True)
            self.log('test/accuracy', metrics['accuracy'], batch_size=batch_size, sync_dist=True)
            self.log('test/top3_accuracy', metrics['top3'], batch_size=batch_size, sync_dist=True)
            self.log('test/top5_accuracy', metrics['top5'], batch_size=batch_size, sync_dist=True)
    
    # def predict_step(self, batch: Any, batch_idx: int) -> Dict:
    #     """Inference on batch from dataloader."""
        
    #     if not (isinstance(batch, dict) and 'batch' in batch):
    #         raise RuntimeError("Data must be batch from dataloader")
        
    #     outputs = self(batch)
        
    #     visible_mask = batch['visible_mask']
    #     log_probs = F.log_softmax(outputs['logits'], dim=-1)
    #     assignments = self.matcher(log_probs, visible_mask)
        
    #     return {
    #         'predictions': assignments['hard_assignments']
    #     }
    
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        opt_config = self.config.optimizer
        sched_config = self.config.scheduler
        
        if opt_config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr = opt_config.learning_rate,
                weight_decay = opt_config.weight_decay,
                betas = opt_config.betas,
                eps = opt_config.eps
            )
        elif opt_config.optimizer_type == "muon":
            # Split parameters for Muon
            muon_params = []
            adamw_params = []
            
            adamw_modules = ['node_encoder', 'edge_encoder', 'classification_head', 'pe_module']
            
            for name, param in self.model.named_parameters():
                use_adamw = False
                
                for module_name in adamw_modules:
                    if module_name in name:
                        use_adamw = True
                        break
                
                # Also use AdamW for layer norms, biases, and 1D parameters
                if 'norm' in name.lower() or 'bias' in name or param.ndim < 2:
                    use_adamw = True
                
                if use_adamw:
                    adamw_params.append(param)
                elif param.ndim == 2:
                    # 2D parameters go to Muon
                    muon_params.append(param)
                else:
                    # Any other parameters go to AdamW
                    adamw_params.append(param)

            # Include Sinkhorn matcher (dustbin) parameters in the optimizer.
            # TODO: Check if we should keep dustbin scores
            # These are scalars / 1D, so they should be handled by the AdamW part of Muon.
            # if self.config.task.target == "canonical":
            #     adamw_params.extend(self.loss_fn.matcher.parameters())

            adamw_lr = opt_config.muon_adamw_lr if opt_config.muon_adamw_lr is not None else opt_config.learning_rate
            
            use_bfloat16 = self.config.system.mixed_precision in ["bf16", "bf16-mixed"]
            
            optimizer = Muon(
                muon_params = muon_params,
                lr = opt_config.muon_lr,
                momentum = opt_config.muon_momentum,
                nesterov = opt_config.muon_nesterov,
                ns_steps = opt_config.muon_ns_steps,
                adamw_params = adamw_params,
                adamw_lr = adamw_lr,
                adamw_betas = opt_config.muon_adamw_betas,
                adamw_eps = opt_config.muon_adamw_eps,
                adamw_wd = opt_config.muon_adamw_wd,
                use_bfloat16 = use_bfloat16
            )
            
            num_muon = len(muon_params)
            num_adamw = len(adamw_params)
            total_muon_params = sum(p.numel() for p in muon_params)
            total_adamw_params = sum(p.numel() for p in adamw_params)
            print(f"Muon optimizer initialized:")
            print(f" - Muon parameters: {num_muon} tensors, {total_muon_params:,} params")
            print(f" - AdamW parameters: {num_adamw} tensors, {total_adamw_params:,} params")
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer_type}")
        
        scheduler = None
        scheduler_config = None
        
        if sched_config.scheduler_type in ["cosine", "cosine_with_warmup"]:
            # Scheduler with warmup + cosine annealing
            warmup_batches = sched_config.warmup_batches
            warmup_start_lr_ratio = sched_config.warmup_start_lr / opt_config.learning_rate
            min_lr_ratio = sched_config.min_lr / opt_config.learning_rate
            
            def lr_lambda(current_step):
                """Linear warmup with Cosine annealing after"""
                
                if current_step < warmup_batches:
                    return warmup_start_lr_ratio + (1.0 - warmup_start_lr_ratio) * (current_step / warmup_batches)
                else:
                    if sched_config.total_training_batches is not None:
                        remaining_batches = sched_config.total_training_batches - warmup_batches
                        progress = min((current_step - warmup_batches) / remaining_batches, 1.0)
                    else:
                        # Decay slowly over a large number of steps if total not specified
                        remaining_batches = 100000
                        progress = min((current_step - warmup_batches) / remaining_batches, 1.0)
                    
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            
            print(f"Learning rate scheduler configured:")
            print(f" - Warmup batches: {warmup_batches}")
            print(f" - Warmup start LR: {sched_config.warmup_start_lr}")
            print(f" - Peak LR: {opt_config.learning_rate}")
            print(f" - Min LR: {sched_config.min_lr}")
            if sched_config.total_training_batches:
                print(f" - Total training batches: {sched_config.total_training_batches}")
            
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',  # Update after each batch
                'frequency': 1
            }
        
        elif sched_config.scheduler_type == "linear":
            # Simple linear decay, no warmup
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor = 1.0,
                end_factor = sched_config.min_lr / opt_config.learning_rate,
                total_iters = sched_config.total_training_batches or 100000
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
            
        elif sched_config.scheduler_type == "constant":
            scheduler = None
            
        if scheduler_config:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_config
            }
        return optimizer
    
    
    def _update_temperature(self):
        """Update Sinkhorn temperature based on training progress"""
        
        if self.trainer.global_step < self.config.sinkhorn.temperature_warmup_steps:
            alpha = self.trainer.global_step / self.config.sinkhorn.temperature_warmup_steps
            temperature = (self.config.sinkhorn.init_temperature * (1 - alpha) + self.config.sinkhorn.final_temperature * alpha)
        else:
            temperature = self.config.sinkhorn.final_temperature
        
        self.loss_fn.matcher.temperature = temperature
        self.matcher.temperature = temperature


def create_trainer(config: Config) -> L.Trainer:
    
    callbacks = [
        ModelCheckpoint(
            dirpath = config.storage.get_checkpoint_path(config.experiment.experiment_name),
            filename = 'epoch={epoch}-val_accuracy={val/accuracy:.4f}',
            monitor = config.experiment.monitor_metric,
            mode = config.experiment.monitor_mode,
            save_top_k = config.experiment.save_top_k,
            save_last = True,
            every_n_train_steps = config.training.checkpoint_interval
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    if config.curriculum.enabled:
        callbacks.append(CurriculumCallback(config.curriculum))
    
    logger = None
    if config.experiment.wandb_project:
        logger = WandbLogger(
            project = config.experiment.wandb_project,
            entity = config.experiment.wandb_entity,
            name = config.experiment.experiment_name,
            tags = config.experiment.wandb_tags,
            config = config.to_dict()
        )
    
    trainer = L.Trainer(
        accelerator = 'gpu' if config.system.device == 'cuda' else 'cpu',
        devices = config.system.num_gpus if config.system.distributed else 1,
        strategy = 'ddp' if config.system.distributed else 'auto',
        precision = config.system.mixed_precision.replace('fp', ''), 
        max_epochs = config.training.epochs,
        gradient_clip_val = config.training.gradient_clip_val,
        accumulate_grad_batches = config.training.accumulation_steps(config.system.num_gpus),
        val_check_interval = config.training.val_check_interval * config.training.accumulation_steps(config.system.num_gpus),
        limit_val_batches = config.training.limit_val_batches,
        log_every_n_steps = config.experiment.log_interval,
        callbacks = callbacks,
        logger = logger,
        enable_checkpointing = True,
        enable_progress_bar = True,
        enable_model_summary = True,
        deterministic = True
    )
    
    return trainer


def train(config: Config, train_loader, val_loader=None, test_loader=None, resume_from=None):
    torch.set_float32_matmul_precision('medium')

    L.seed_everything(config.training.seed, workers=True)
    
    if resume_from:
        print(f"Loading model from checkpoint: {resume_from}")
        model = LightningModel.load_from_checkpoint(resume_from, config=config)
    else:
        model = LightningModel(config)
    
    trainer = create_trainer(config)
    
    # trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from)
    # TODO: This now restarts the optimizer state, which is not "true finetuning." It is used for superglue finetuning, but it is not how original finetuning worked!
    trainer.fit(model, train_loader, val_loader) # I gotta check if this is the right way to go!
    
    if test_loader:
        trainer.test(model, test_loader)
    
    return model, trainer
