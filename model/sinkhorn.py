import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import math


def log_sinkhorn_iterations(log_scores, log_mu, log_nu, iters: int = 5):
    """Stabilized Sinkhorn iterations in log-space.
    
    Args:
        log_scores: [B, N, M] log-scaled similarity scores
        log_mu: [B, N] source marginals in log space  
        log_nu: [B, M] target marginals in log space
        iters: Number of iterations
    
    Returns:
        Soft assignment matrix [B, N, M]
    """
    
    dtype = log_scores.dtype # float32
    log_scores = log_scores.float()
    log_mu = log_mu.float()
    log_nu = log_nu.float()
    
    B, N, M = log_scores.shape
    log_u = torch.zeros(B, N, device=log_scores.device)
    log_v = torch.zeros(B, M, device=log_scores.device)
    
    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(log_scores + log_v.unsqueeze(1), dim=-1) # Row normalization
        log_v = log_nu - torch.logsumexp(log_scores + log_u.unsqueeze(-1), dim=1) # Col normalization  
    
    # final assignment matrix
    P = torch.exp(log_scores + log_u.unsqueeze(-1) + log_v.unsqueeze(1))
    
    return P.to(dtype)


class Matcher(nn.Module):
    """Sinkhorn matcher for assigning N observed nodes to M canonical IDs."""
    
    def __init__(self, num_canonical: int = 558, num_iterations: int = 5, temperature: float = 1.0, dustbin_weight: float = 1.0, epsilon: float = 1e-8, one_to_one: bool = False):
        super().__init__()
        self.num_canonical = num_canonical
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.epsilon = epsilon
        self.one_to_one = one_to_one 
        
        self.dustbin_score = nn.Parameter(torch.tensor(-1.0)) # Learnable dustbin scores
        self.dustbin_weight = dustbin_weight
    
    
    @torch.amp.autocast('cuda', enabled=False) # TODO fix ts to make it more efficient, figure out why torch compile not working 
    def forward(self, log_probs: torch.Tensor, visible_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute soft assignments from observed nodes to canonical IDs.
        
        Args:
            log_probs: [B, N, 558] log probabilities for each node's canonical ID
            visible_mask: [B, N] mask indicating which nodes are visible
        
        Returns:
            Dictionary containing:
             - 'soft_assignments': [B, N, 558] soft assignment matrix
             - 'hard_assignments': [B, N] hard assignments (argmax)
        """
        
        B, N, C = log_probs.shape
        
        if C != self.num_canonical:
            raise ValueError(f"Expected {self.num_canonical} classes, got {C}")
        
        device = log_probs.device
        
        if visible_mask.shape != (B, N):
            raise ValueError(f"Visible_mask should have shape {(B, N)}, but got {visible_mask.shape}")
        
        log_scores = log_probs / self.temperature
        
        # dustbin for unmatched nodes
        dustbin_score = self.dustbin_score / self.temperature
        dustbin_col = dustbin_score.expand(B, N, 1)
        log_scores_with_dustbin = torch.cat([log_scores, dustbin_col], dim=-1)
        
        # Source marginals: uniform over visible nodes
        log_mu = torch.full((B, N), float('-inf'), device=device)
        for b in range(B):
            visible_nodes = visible_mask[b]
            num_visible = visible_nodes.sum().item()
            if num_visible > 0:
                log_mu[b, visible_nodes] = - math.log(num_visible)
        
        # Target marginals:
        if self.one_to_one:
            # Bias towards one-to-one matching
            log_nu = torch.full((B, C + 1), float('-inf'), device=device)
            
            for b in range(B):
                num_visible = visible_mask[b].sum().item()
                
                if num_visible <= C:
                    canonical_mass = 1.0 / C
                    dustbin_mass = max(1e-8, (C - num_visible) / C)  # Dustbin gets remaining mass (C - num_visible)/C, avoid log(0)
                else:
                    canonical_mass = 1.0 / num_visible
                    dustbin_mass = (num_visible - C) / num_visible
                
                log_nu[b, :C] = math.log(canonical_mass)
                log_nu[b, C] = math.log(dustbin_mass)  # dustbin column
        else:
            # Many-to-one matching: allow multiple assignments to same canonical ID
            log_nu = torch.full((B, C + 1), math.log(1.0 / (C + 1)), device=device)
        
        # Run Sinkhorn iterations
        soft_assignments = log_sinkhorn_iterations(log_scores_with_dustbin, log_mu, log_nu, self.num_iterations)
        soft_assignments = soft_assignments[:, :, :-1]
        
        hard_assignments = torch.argmax(soft_assignments, dim=-1)
        
        return {
            'soft_assignments': soft_assignments,
            'hard_assignments': hard_assignments,
            'dustbin_weights': soft_assignments[:, :, -1:]
        }


class AssignmentLoss(nn.Module):
    """Loss function for canonical ID assignment using Sinkhorn"""
    
    def __init__(self, num_canonical: int = 558, num_iterations: int = 20, temperature: float = 1.0, class_weight: float = 1.0, assignment_weight: float = 0.5, label_smoothing: float = 0.0, worm_averaged_loss: bool = True, one_to_one: bool = False):

        super().__init__()
        
        self.matcher = Matcher(num_canonical=num_canonical, num_iterations=num_iterations, temperature=temperature, one_to_one=one_to_one)
        
        self.num_canonical = num_canonical
        self.class_weight = class_weight
        self.assignment_weight = assignment_weight
        self.label_smoothing = label_smoothing
        self.worm_averaged_loss = worm_averaged_loss
    
    
    @torch.amp.autocast('cuda', dtype=torch.bfloat16)
    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute loss for canonical ID assignment.
        
        Args:
            outputs: Dictionary with 'logits' [B, N, 558]
            targets: Dictionary with 'labels' [B, max_nodes] and 'visible_mask' [B, N]
        
        Returns:
            Dictionary of losses
        """
        
        log_probs = F.log_softmax(outputs['logits'], dim=-1)
        
        with torch.amp.autocast('cuda', enabled=False):
            assignments = self.matcher(log_probs.float(), targets['visible_mask'])
        
        soft_assignments = assignments['soft_assignments']
        
        losses = {}
        
        class_loss = self._classification_loss(log_probs, targets)
        losses['class_loss'] = class_loss
        
        assignment_loss = self._assignment_confidence_loss(soft_assignments, targets)
        losses['assignment_loss'] = assignment_loss
        
        total_loss = (self.class_weight * losses['class_loss'] + self.assignment_weight * losses['assignment_loss'])
        
        losses['total_loss'] = total_loss
        losses['soft_assignments'] = soft_assignments
        
        return losses
    
    
    def _compute_single_batch_loss(self, log_probs: torch.Tensor, labels: torch.Tensor, visible_indices: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:

        if len(visible_indices) == 0:
            return None
        
        visible_log_probs = log_probs[visible_indices]
        visible_targets = labels[:len(visible_indices)]
        
        if self.label_smoothing > 0:
            num_classes = visible_log_probs.shape[-1]
            smooth_targets = torch.full_like(visible_log_probs, self.label_smoothing / num_classes)
            smooth_targets.scatter_(-1, visible_targets.unsqueeze(-1), 1.0 - self.label_smoothing)
            
            if reduction == 'mean':
                return -(smooth_targets * visible_log_probs).sum() / len(visible_indices)
            else:
                return -(smooth_targets * visible_log_probs).sum()
        else:
            return F.nll_loss(visible_log_probs, visible_targets, reduction=reduction)
    
    
    def _classification_loss(self, log_probs: torch.Tensor, targets: Dict) -> torch.Tensor:
        """Cross-entropy loss for visible nodes computed either per-node (default) or per-worm averaged."""
        
        batch_size = log_probs.shape[0]
        device = log_probs.device
        
        if self.worm_averaged_loss:
            worm_losses = []
            
            for b in range(batch_size):
                visible_mask = targets['visible_mask'][b]
                visible_indices = torch.where(visible_mask)[0]
                
                loss = self._compute_single_batch_loss(log_probs[b], targets['labels'][b], visible_indices, reduction='mean')

                if loss is not None:
                    worm_losses.append(loss)
            
            if len(worm_losses) == 0:
                raise ValueError("No worms found in batch for classification loss.")
            
            # Average across worms (each worm contributes equally)
            return torch.stack(worm_losses).mean()
        
        else:
            total_loss = torch.tensor(0.0, device=device)
            total_nodes = 0
            
            for b in range(batch_size):
                visible_mask = targets['visible_mask'][b]
                visible_indices = torch.where(visible_mask)[0]
                
                loss = self._compute_single_batch_loss(log_probs[b], targets['labels'][b], visible_indices, reduction='sum')

                if loss is not None:
                    total_loss = total_loss + loss
                    total_nodes += len(visible_indices)
            
            # Node-averaged loss
            return total_loss / max(total_nodes, 1)
    
    
    def _compute_single_batch_entropy(self, soft_assignments: torch.Tensor, visible_indices: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        if len(visible_indices) == 0:
            return None
        
        visible_assignments = soft_assignments[visible_indices]
        
        eps = 1e-8
        entropy = -(visible_assignments * torch.log(visible_assignments + eps)).sum(dim=-1)
        
        if reduction == 'mean':
            return entropy.mean()
        else:
            return entropy.sum()
    
    def _assignment_confidence_loss(self, soft_assignments: torch.Tensor, targets: Dict) -> torch.Tensor:
        """Encourage confident assignments for visible nodes."""
        
        batch_size = soft_assignments.shape[0]
        device = soft_assignments.device
        
        if self.worm_averaged_loss:
            worm_entropies = []
            
            for b in range(batch_size):
                visible_mask = targets['visible_mask'][b]
                visible_indices = torch.where(visible_mask)[0]

                entropy = self._compute_single_batch_entropy(soft_assignments[b], visible_indices, reduction='mean')

                if entropy is not None:
                    worm_entropies.append(entropy)
            
            if len(worm_entropies) > 0:
                return torch.stack(worm_entropies).mean()
            else:
                return torch.tensor(0.0, device=device)
        
        else:
            total_entropy = torch.tensor(0.0, device=device)
            total_nodes = 0
            
            for b in range(batch_size):
                visible_mask = targets['visible_mask'][b]
                visible_indices = torch.where(visible_mask)[0]

                entropy = self._compute_single_batch_entropy(soft_assignments[b], visible_indices, reduction='sum')

                if entropy is not None:
                    total_entropy = total_entropy + entropy
                    total_nodes += len(visible_indices)
            
            return total_entropy / max(total_nodes, 1)


def compute_assignment_accuracy(outputs: Dict, targets: Dict, assignments: torch.Tensor) -> Dict[str, float]:
    """Compute accuracy metrics using Sinkhorn assignments."""
    
    batch_size = outputs['logits'].shape[0]
    
    worm_exact_accs = []
    worm_top3_accs = []
    worm_top5_accs = []
    total_nodes_processed = 0
    
    for b in range(batch_size):
        visible_mask = targets['visible_mask'][b]
        visible_indices = torch.where(visible_mask)[0]
        
        if len(visible_indices) == 0:
            continue
        
        visible_assignments = assignments[b, visible_indices]  # [num_visible, 558]
        visible_targets = targets['labels'][b][:len(visible_indices)]
        
        predicted_ids = torch.argmax(visible_assignments, dim=-1)
        
        top5_preds = torch.topk(visible_assignments, k=min(5, visible_assignments.shape[-1]), dim=-1).indices
        top3_preds = top5_preds[:, :3]
        
        num_nodes = len(visible_indices)
        worm_correct = (predicted_ids == visible_targets).sum().item()
        worm_exact_acc = worm_correct / num_nodes
        worm_exact_accs.append(worm_exact_acc)
        
        worm_top3_correct = (visible_targets.unsqueeze(1) == top3_preds).any(dim=1).sum().item()
        worm_top3_acc = worm_top3_correct / num_nodes
        worm_top3_accs.append(worm_top3_acc)
        
        worm_top5_correct = (visible_targets.unsqueeze(1) == top5_preds).any(dim=1).sum().item()
        worm_top5_acc = worm_top5_correct / num_nodes
        worm_top5_accs.append(worm_top5_acc)
        
        total_nodes_processed += num_nodes
    
    if len(worm_exact_accs) > 0:
        mean_exact_acc = sum(worm_exact_accs) / len(worm_exact_accs)
        mean_top3_acc = sum(worm_top3_accs) / len(worm_top3_accs)
        mean_top5_acc = sum(worm_top5_accs) / len(worm_top5_accs)
    else:
        mean_exact_acc = 0.0
        mean_top3_acc = 0.0
        mean_top5_acc = 0.0
    
    return {
        'exact_accuracy': mean_exact_acc,
        'top3_accuracy': mean_top3_acc,
        'top5_accuracy': mean_top5_acc,
        'total_nodes': total_nodes_processed,
        'num_worms': len(worm_exact_accs)
    }
