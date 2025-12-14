import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import math
from scipy.optimize import linear_sum_assignment


def log_sinkhorn_iterations(scaled_scores, log_mu, log_nu, iters: int = 5):
    """Stabilized Sinkhorn iterations in log-space.
    
    Args:
        scaled_scores: [B, N, M] log-scaled similarity scores
        log_mu: [B, N] source marginals in log space  
        log_nu: [B, M] target marginals in log space
        iters: Number of iterations
    
    Returns:
        Soft assignment matrix [B, N, M]
    """
    
    dtype = scaled_scores.dtype # float32
    scaled_scores = scaled_scores.float()
    log_mu = log_mu.float()
    log_nu = log_nu.float()
    
    B, N, M = scaled_scores.shape
    log_u = torch.zeros(B, N, device=scaled_scores.device)
    log_v = torch.zeros(B, M, device=scaled_scores.device)
    
    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(scaled_scores + log_v.unsqueeze(1), dim=-1) # Row normalization
        log_v = log_nu - torch.logsumexp(scaled_scores + log_u.unsqueeze(-1), dim=1) # Col normalization  
    
    # final assignment matrix
    P = torch.exp(scaled_scores + log_u.unsqueeze(-1) + log_v.unsqueeze(1))
    
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
        
        # Both row and column dustbins
        # TODO: Understand whether or not we should have separate learnable dustbin scores
        self.dustbin_col_score = nn.Parameter(torch.tensor(-1.0))
        self.dustbin_weight = dustbin_weight

        if self.one_to_one:
            self.dustbin_row_score = nn.Parameter(torch.tensor(-1.0))
            self.dustbin_corner_score = nn.Parameter(torch.tensor(-1.0))

        else:
            self.dustbin_row_score = None  # or register_buffer(..., persistent=False)
            self.dustbin_corner_score = None

    # TODO: This is only used to load legacy models; can be removed later.
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        legacy_key = prefix + "dustbin_score"
        col_key = prefix + "dustbin_col_score"
        if legacy_key in state_dict and col_key not in state_dict:
            state_dict[col_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.amp.autocast('cuda', enabled=False) # TODO fix ts to make it more efficient, figure out why torch compile not working 
    def forward(self, scores: torch.Tensor, visible_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute soft assignments from observed nodes to canonical IDs.
        
        Args:
            scores: [B, N, 558] similarity scores (raw logits or log probabilities)
            visible_mask: [B, N] mask indicating which nodes are visible
        
        Returns:
            Dictionary containing:
             - 'soft_assignments': [B, N, 558] soft assignment matrix
             - 'hard_assignments': [B, N] hard assignments (argmax)
        """
        
        B, N, C = scores.shape
        
        if C != self.num_canonical:
            raise ValueError(f"Expected {self.num_canonical} classes, got {C}")
        
        device = scores.device
        
        if visible_mask.shape != (B, N):
            raise ValueError(f"Visible_mask should have shape {(B, N)}, but got {visible_mask.shape}")
        
        scaled_scores = scores / self.temperature

        if self.one_to_one:

            # Build (N+1) x (C+1) score matrix: real+ dustbin column and row.
            dustbin_col = (self.dustbin_col_score / self.temperature).expand(B, N, 1)
            aug_scores = torch.cat([scaled_scores, dustbin_col], dim=-1)   # [B, N, C+1]

            dustbin_row = (self.dustbin_row_score / self.temperature).expand(B, 1, C)
            dustbin_corner = (self.dustbin_corner_score / self.temperature).expand(B, 1, 1)
            aug_scores = torch.cat(
                [aug_scores, torch.cat([dustbin_row, dustbin_corner], dim=-1)], dim=1)   # [B, N+1, C+1]

            # SuperGlue-style marginals in log-space, with padding support.
            # Shape: rows = N real rows + 1 dustbin row; cols = C real cols + 1 dustbin col.
            log_mu = torch.full((B, N + 1), float("-inf"), device=device, dtype=aug_scores.dtype)
            log_nu = torch.full((B, C + 1), float("-inf"), device=device, dtype=aug_scores.dtype)
            # TODO: I could put this in a separate function.
            for b in range(B):
                visible = visible_mask[b]                      # [N]
                visible_idx = visible.nonzero(as_tuple=False).squeeze(-1)  # indices of real nodes
                m = int(visible_idx.numel())                   # #visible nodes for this worm
                n = C                                          # #canonical IDs

                if m == 0:
                    # No visible nodes: keep all marginals -inf (no mass); Sinkhorn will yield zeros.
                    continue

                # Follow SuperGlue: μ = [1,...,1,n], ν = [1,...,1,m] up to a shared shift.
                one = aug_scores.new_tensor(1.0)              # scalar with same device/dtype
                ms = m * one                                  # tensor(m)
                ns = n * one                                  # tensor(n)
                norm = -(ms + ns).log()                       # = -log(m+n), scalar

                # --- Row marginals μ (length N+1) ---
                # Visible rows: log(1) - log(m+n) = norm
                log_mu[b, visible_idx] = norm

                # Dustbin row (index N): log(n) - log(m+n)
                log_mu[b, N] = ns.log() + norm

                # Padded rows (non-visible, not dustbin) remain -inf → μ_i = 0 → P row = 0.

                # --- Column marginals ν (length C+1) ---
                # Real canonical IDs: log(1) - log(m+n) = norm
                log_nu[b, :C] = norm

                # Dustbin column (index C): log(m) - log(m+n)
                log_nu[b, C] = ms.log() + norm

            # Run log-space Sinkhorn as before.
            full_assignments = log_sinkhorn_iterations(
                aug_scores, log_mu, log_nu, self.num_iterations
            )

            # Extract the blocks:
            soft_assignments = full_assignments[:, :N, :C]       # real rows × real cols
            dustbin_assignments = full_assignments[:, :N, C:C+1] # real rows × dustbin col
            dustbin_row_assignments = full_assignments[:, N:N+1, :C]  # dustbin row × real cols
        #TODO: I am not sure if this branch still works properly
        else:
            # Many-to-one: keep original single-column dustbin.
            dustbin_col = (self.dustbin_col_score / self.temperature).expand(B, N, 1)
            aug_scores = torch.cat([scaled_scores, dustbin_col], dim=-1)

            log_mu = torch.full((B, N), float('-inf'), device=device)
            for b in range(B):
                visible = visible_mask[b]
                num_visible = int(visible.sum())
                if num_visible > 0:
                    log_mu[b, visible] = -math.log(num_visible)

            log_nu = torch.full((B, C + 1), math.log(1.0 / (C + 1)), device=device)

            full_assignments = log_sinkhorn_iterations(aug_scores, log_mu, log_nu, self.num_iterations)
            soft_assignments = full_assignments[:, :, :-1]
            dustbin_assignments = full_assignments[:, :, -1:]
            dustbin_row_assignments = None

        hard_assignments = torch.argmax(soft_assignments, dim=-1)
        
        outputs = {
            'full_assignments': full_assignments,
            'soft_assignments': soft_assignments,
            'hard_assignments': hard_assignments,
            'dustbin_weights': dustbin_assignments,
            'dustbin_row_weights': dustbin_row_assignments,
        }

        return outputs

class AssignmentLoss(nn.Module):
    """Loss function for canonical ID assignment using Sinkhorn"""
    
    def __init__(self, num_canonical: int = 558, num_iterations: int = 20, temperature: float = 1.0, class_weight: float = 1.0, assignment_weight: float = 0.5, 
                 label_smoothing: float = 0.0, worm_averaged_loss: bool = True, one_to_one: bool = False, use_superglue_loss: bool = False,
                 use_topk_sinkhorn_mask: bool = False, topk_sinkhorn_k: int = 5):

        super().__init__()
        
        self.matcher = Matcher(num_canonical=num_canonical, num_iterations=num_iterations, temperature=temperature, one_to_one=one_to_one)
        
        self.num_canonical = num_canonical
        self.class_weight = class_weight
        self.assignment_weight = assignment_weight
        self.label_smoothing = label_smoothing
        self.worm_averaged_loss = worm_averaged_loss
        self.use_superglue_loss = use_superglue_loss
        # New flags for top-K masking
        self.use_topk_sinkhorn_mask = use_topk_sinkhorn_mask
        self.topk_sinkhorn_k = topk_sinkhorn_k
    
    
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
        
        logits = outputs['logits']
        log_probs = F.log_softmax(logits, dim=-1)

        # If we are using top-K masking, we always supervise with SuperGlue loss.
        effective_use_superglue = self.use_superglue_loss or self.use_topk_sinkhorn_mask

        # For SuperGlue (or top-K mode), run Sinkhorn on raw logits; otherwise use log-probs.
        # Decide which scores go into the matcher (raw logits vs log-probs)
        sinkhorn_scores = logits if effective_use_superglue else log_probs

        # Optional top-K masking on the canonical-ID dimension, per row
        #TODO: I should move this to a separate function
        if self.use_topk_sinkhorn_mask:
            B, N, C = sinkhorn_scores.shape
            K = min(self.topk_sinkhorn_k, C)
            device = sinkhorn_scores.device

            # Start with all positions masked out
            topk_mask = torch.zeros((B, N, C), device=device, dtype=torch.bool)

            visible_mask = targets['visible_mask']  # [B, N]
            # Use logits for defining top-K, as discussed
            with torch.no_grad():
                for b in range(B):
                    vis = visible_mask[b].nonzero(as_tuple=False).squeeze(-1)
                    if vis.numel() == 0:
                        continue
                    # [Nv, C]
                    logits_b = logits[b, vis]
                    _, topk_idx = torch.topk(logits_b, k=K, dim=-1)
                    
                    # rows: [Nv, K], cols: [Nv, K]
                    rows = vis.unsqueeze(1).expand(-1, K)
                    topk_mask[b, rows, topk_idx] = True

            # Set non-top-K scores to a very negative value so they get zero probability after Sinkhorn
            very_neg = -1e9
            sinkhorn_scores = sinkhorn_scores.masked_fill(~topk_mask, very_neg)

        with torch.amp.autocast('cuda', enabled=False):
            assignments = self.matcher(sinkhorn_scores.float(), targets['visible_mask'])

        soft_assignments = assignments['soft_assignments']
        losses = {}

        if 'full_assignments' in assignments:
            losses['full_assignments'] = assignments['full_assignments']

        if effective_use_superglue:
            if not self.matcher.one_to_one:
                raise ValueError("SuperGlue loss requires one_to_one=True for row+col dustbins.")
            full_assignments = assignments.get('full_assignments')
            if full_assignments is None:
                raise ValueError("Matcher must return full_assignments for SuperGlue loss.")
            superglue_loss = self._superglue_loss(full_assignments, targets['visible_mask'], targets['labels'])
            losses['superglue_loss'] = superglue_loss
            total_loss = superglue_loss
        else:
            class_loss = self._classification_loss(log_probs, targets)
            assignment_loss = self._assignment_confidence_loss(soft_assignments, targets)
            losses['class_loss'] = class_loss
            losses['assignment_loss'] = assignment_loss
            total_loss = self.class_weight * class_loss + self.assignment_weight * assignment_loss

        losses['total_loss'] = total_loss
        losses['soft_assignments'] = soft_assignments
        losses['dustbin_weights'] = assignments['dustbin_weights']
        if 'dustbin_row_weights' in assignments:
            losses['dustbin_row_weights'] = assignments['dustbin_row_weights']

        return losses
    
    def _superglue_loss(self, full_P, visible_mask, labels):
        """
        full_P: [B, N+1, C+1]
        visible_mask: [B, N]
        labels: [B, N]
        """
        B, N_plus, C_plus = full_P.shape
        N = N_plus - 1
        C = C_plus - 1
        dustbin_row = N
        dustbin_col = C
        eps = 1e-12

        device = full_P.device
        dtype = full_P.dtype

        # Normalize by m + C per batch
        scale = torch.ones(B, 1, 1, device=device, dtype=dtype)
        for b in range(B):
            m = int(visible_mask[b].sum().item())
            if m > 0:
                scale[b, 0, 0] = m + C
            else:
                scale[b, 0, 0] = 1.0

        P_scaled = full_P * scale
        log_P = (P_scaled.clamp_min(eps)).log()

        # Debugging
        
        with torch.no_grad():
            b = 0  # pick a batch element to inspect
            if B > 0:
                P_b = P_scaled[b]              # [N+1, C+1]
                row_sums = P_b.sum(dim=-1)     # [N+1]
                col_sums = P_b.sum(dim=-2)     # [C+1]
                total_mass = P_b.sum()

                vis = visible_mask[b].bool()   # [N]
                N_plus = row_sums.shape[0]
                N = N_plus - 1                 # dustbin row index
                vis_rows = vis.nonzero(as_tuple=False).squeeze(-1)
                nonvis_rows = (~vis).nonzero(as_tuple=False).squeeze(-1)

                print("=== SuperGlue P_scaled debug ===", flush=True)
                print("m (visible rows):", int(vis.sum().item()), flush=True)
                print("C (real columns):", C, flush=True)

                # Visible vs non-visible rows
                if vis_rows.numel() > 0:
                    print("visible row_sums:", row_sums[vis_rows], flush=True)
                else:
                    print("visible row_sums: []", flush=True)

                if nonvis_rows.numel() > 0:
                    print("non-visible row_sums:", row_sums[nonvis_rows], flush=True)
                else:
                    print("non-visible row_sums: []", flush=True)

                print("dustbin row_sum:", row_sums[N], flush=True)

                # Columns (first few + dustbin)
                print("first 10 col_sums:", col_sums[:10], flush=True)
                print("dustbin col_sum:", col_sums[C], flush=True)

                # Example row values and total mass
                print("P_scaled[0, :10]:", P_b[0, :10], flush=True)
                print("P_scaled total_mass:", total_mass, flush=True)
        # --- end debug block ---
            

        # Two aggregation modes:
        # - worm_averaged_loss == False:
        #     element-averaged over all batches (original behavior).
        # - worm_averaged_loss == True:
        #     compute a mean over all elements for each worm, then average worms.
        all_nll = []    # list of per-worm tensors (for element-averaged mode)
        worm_means = [] # list of per-worm scalars (for worm-averaged mode)

        for b in range(B):
            vis = visible_mask[b].bool()
            vis_idx = vis.nonzero(as_tuple=False).squeeze(-1)

            if vis_idx.numel() == 0:
                continue

            labels_b = labels[b, vis_idx]

            # Collect this worm's contributions in a list
            worm_terms = []

            # Matched pairs
            matched_mask = labels_b >= 0
            if matched_mask.any():
                matched_rows = vis_idx[matched_mask]
                matched_cols = labels_b[matched_mask]
                worm_terms.append(-log_P[b, matched_rows, matched_cols])

            # Unmatched rows → dustbin column
            unmatched_mask = labels_b < 0
            if unmatched_mask.any():
                unmatched_rows = vis_idx[unmatched_mask]
                worm_terms.append(-log_P[b, unmatched_rows, dustbin_col])

            # Unused canonical IDs → dustbin row
            used_mask = torch.zeros(C, device=device, dtype=torch.bool)
            if matched_mask.any():
                used_mask.scatter_(0, labels_b[matched_mask], True)
            unused_cols = (~used_mask).nonzero(as_tuple=False).squeeze(-1)
            if unused_cols.numel() > 0:
                worm_terms.append(-log_P[b, dustbin_row, unused_cols])

            # Aggregate this worm's contributions
            if worm_terms:
                worm_nll = torch.cat(worm_terms)
                if self.worm_averaged_loss:
                    # Worm-averaged: each worm contributes one scalar = mean over its own terms.
                    worm_means.append(worm_nll.mean())
                else:
                    # Original behavior: element-averaged across the whole batch.
                    all_nll.append(worm_nll)

        if self.worm_averaged_loss:
            if worm_means:
                return torch.stack(worm_means).mean()
            else:
                return torch.tensor(0.0, device=device)
        else:
            if all_nll:
                return torch.cat(all_nll).mean()
            else:
                return torch.tensor(0.0, device=device)
        
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

# TODO: I have to check this implementation and (maybe) move it elsewhere.        
def hungarian_match_from_logits(logits: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
    """
    One-to-one matching via Hungarian. Assumes every visible row must match some canonical ID.
    logits: [B, N, C]; visible_mask: [B, N]
    Returns [B, N] with canonical IDs; padded rows are -1.
    """
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian matching")
    B, N, C = logits.shape
    device = logits.device
    assignments = torch.full((B, N), -1, device=device, dtype=torch.long)
    for b in range(B):
        vis = visible_mask[b].nonzero(as_tuple=False).squeeze(-1)
        if vis.numel() == 0:
            continue
        scores = logits[b, vis].float()                # [Nv, C], higher is better
        cost = (-scores).detach().cpu().numpy()  # Hungarian minimizes
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = torch.as_tensor(col_ind, device=device, dtype=torch.long)
        assignments[b, vis[row_ind]] = assigned
    return assignments

def _compute_superglue_metrics(full_assignments: torch.Tensor,
                               visible_mask: torch.Tensor,
                               labels: torch.Tensor) -> Dict[str, float]:
    """Compute detailed metrics for SuperGlue-style matching."""
    B, N_plus, C_plus = full_assignments.shape
    N = N_plus - 1
    C = C_plus - 1
    dustbin_row = N
    dustbin_col = C
    eps = 1e-12

    device = full_assignments.device

    total_matched_pairs = 0
    total_unmatched_rows = 0
    total_unmatched_cols = 0

    sum_matched_conf = 0.0
    sum_unmatched_row_conf = 0.0
    sum_unmatched_col_conf = 0.0

    sum_loss_matched = 0.0
    sum_loss_unmatched_rows = 0.0
    sum_loss_unmatched_cols = 0.0

    num_correct_matches = 0

    for b in range(B):
        vis = visible_mask[b].bool()
        vis_idx = vis.nonzero(as_tuple=False).squeeze(-1)

        if vis_idx.numel() == 0:
            continue

        m = int(vis_idx.numel())
        K = m + C

        # Same K-scaling idea as in _superglue_loss
        P_norm = full_assignments[b] * K
        log_P = (P_norm.clamp_min(eps)).log()

        labels_b = labels[b, vis_idx]

        # Matched pairs (labels >= 0)
        matched_mask = labels_b >= 0
        if matched_mask.any():
            matched_rows = vis_idx[matched_mask]
            matched_cols = labels_b[matched_mask]

            # Argmax over real columns only (0..C-1)
            sinkhorn_preds = full_assignments[b, matched_rows, :C].argmax(dim=-1)
            num_correct_matches += (sinkhorn_preds == matched_cols).sum().item()

            matched_confs = P_norm[matched_rows, matched_cols]
            sum_matched_conf += matched_confs.sum().item()
            sum_loss_matched += (-log_P[matched_rows, matched_cols]).sum().item()
            total_matched_pairs += len(matched_rows)

        # Unmatched rows → dustbin column (labels < 0)
        unmatched_mask = labels_b < 0
        if unmatched_mask.any():
            unmatched_rows = vis_idx[unmatched_mask]
            unmatched_row_confs = P_norm[unmatched_rows, dustbin_col]
            sum_unmatched_row_conf += unmatched_row_confs.sum().item()
            sum_loss_unmatched_rows += (-log_P[unmatched_rows, dustbin_col]).sum().item()
            total_unmatched_rows += len(unmatched_rows)

        # Unused canonical IDs → dustbin row
        used_mask = torch.zeros(C, device=device, dtype=torch.bool)
        if matched_mask.any():
            used_mask.scatter_(0, labels_b[matched_mask], True)

        unused_cols = (~used_mask).nonzero(as_tuple=False).squeeze(-1)
        if unused_cols.numel() > 0:
            unmatched_col_confs = P_norm[dustbin_row, unused_cols]
            sum_unmatched_col_conf += unmatched_col_confs.sum().item()
            sum_loss_unmatched_cols += (-log_P[dustbin_row, unused_cols]).sum().item()
            total_unmatched_cols += len(unused_cols)

    metrics = {
        # Accuracy on rows labeled as matched (labels >= 0)
        'sg_exact_match_acc': num_correct_matches / max(total_matched_pairs, 1),

        # Counts
        'sg_num_matched': total_matched_pairs,
        'sg_num_unmatched_rows': total_unmatched_rows,
        'sg_num_unmatched_cols': total_unmatched_cols,

        # Confidence (scaled probabilities)
        'sg_conf_matched': sum_matched_conf / max(total_matched_pairs, 1),
        'sg_conf_unmatched_rows': sum_unmatched_row_conf / max(total_unmatched_rows, 1),
        'sg_conf_unmatched_cols': sum_unmatched_col_conf / max(total_unmatched_cols, 1),

        # Mean NLL per element in each category
        'sg_loss_matched': sum_loss_matched / max(total_matched_pairs, 1),
        'sg_loss_unmatched_rows': sum_loss_unmatched_rows / max(total_unmatched_rows, 1),
        'sg_loss_unmatched_cols': sum_loss_unmatched_cols / max(total_unmatched_cols, 1),
    }

    return metrics

# TODO: This has to be separated into multiple function
def compute_assignment_accuracy(outputs: Dict,
                                targets: Dict,
                                assignments: torch.Tensor,
                                dustbin_weights: Optional[torch.Tensor] = None,
                                dustbin_row_weights: Optional[torch.Tensor] = None,
                                compute_hungarian: bool = False,
                                full_assignments: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Compute accuracy metrics using Sinkhorn assignments.

    Optionally, if full_assignments is provided (one-to-one / SuperGlue mode),
    also compute SuperGlue-specific diagnostics.
    """

    hungarian_assignments = None
    if compute_hungarian:
        hungarian_assignments = hungarian_match_from_logits(outputs['logits'], targets['visible_mask'])

    probabilities = F.softmax(outputs['logits'], dim=-1)
    batch_size = outputs['logits'].shape[0]

    worm_exact_accs: List[float] = []
    worm_top3_accs: List[float] = []
    worm_top5_accs: List[float] = []
    total_nodes_processed = 0

    worm_exact_accs_logits: List[float] = []
    worm_top3_accs_logits: List[float] = []
    worm_top5_accs_logits: List[float] = []

    sum_top1_mass = 0.0
    sum_top2_mass = 0.0
    sum_top3_mass = 0.0
    sum_top5_mass = 0.0
    sum_top10_mass = 0.0
    sum_gap_top1_correct = 0.0
    sum_rank_top1_correct = 0.0

    sum_top1_mass_sinkhorn = 0.0
    sum_top2_mass_sinkhorn = 0.0
    sum_top3_mass_sinkhorn = 0.0
    sum_top5_mass_sinkhorn = 0.0
    sum_top10_mass_sinkhorn = 0.0
    sum_gap_top1_correct_sinkhorn = 0.0
    sum_rank_top1_correct_sinkhorn = 0.0

    sum_dustbin_share = 0.0
    mispredicted_nodes = 0
    sum_dustbin_row_share = 0.0
    total_canonical_slots = 0

    worm_exact_accs_hungarian: List[float] = []

    # squeeze the last dimension once so we can index easily
    if dustbin_weights is not None:
        dustbin_weights = dustbin_weights.squeeze(-1)

    if dustbin_row_weights is not None:
        dustbin_row_weights = dustbin_row_weights.squeeze(1)

    for b in range(batch_size):
        visible_mask = targets['visible_mask'][b]
        visible_indices = torch.where(visible_mask)[0]

        if len(visible_indices) == 0:
            continue

        visible_assignments = assignments[b, visible_indices]  # [num_visible, 558]
        row_sum_canon = visible_assignments.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        visible_assignments_norm = visible_assignments / row_sum_canon  # rows now sum to 1 across canonical IDs

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

        visible_probs = probabilities[b, visible_indices]
        max_k = min(10, visible_probs.size(-1))
        topk_softmax, _ = torch.topk(visible_probs, k=max_k, dim=-1)

        top1_prob = topk_softmax[:, 0]
        top2_prob = topk_softmax[:, :min(2, max_k)].sum(dim=-1)
        top3_mass = topk_softmax[:, :min(3, max_k)].sum(dim=-1)
        top5_mass = topk_softmax[:, :min(5, max_k)].sum(dim=-1)
        top10_mass = topk_softmax.sum(dim=-1)

        # --- logits-only predictions (no Sinkhorn) ---
        logits_topk_vals, logits_topk_idx = torch.topk(visible_probs, k=5, dim=-1)
        logits_pred_top1 = logits_topk_idx[:, 0]

        correct_logits = (logits_pred_top1 == visible_targets)
        logits_top1_acc = correct_logits.float().mean().item()

        # top-3 / top-5 using logits
        logits_in_top3 = (visible_targets.unsqueeze(1) == logits_topk_idx[:, :3]).any(dim=1)
        logits_top3_acc = logits_in_top3.float().mean().item()

        logits_in_top5 = (visible_targets.unsqueeze(1) == logits_topk_idx[:, :5]).any(dim=1)
        logits_top5_acc = logits_in_top5.float().mean().item()

        worm_exact_accs_logits.append(logits_top1_acc)
        worm_top3_accs_logits.append(logits_top3_acc)
        worm_top5_accs_logits.append(logits_top5_acc)

        ##########################################################

        sum_top1_mass += top1_prob.sum().item()
        sum_top2_mass += top2_prob.sum().item()
        sum_top3_mass += top3_mass.sum().item()
        sum_top5_mass += top5_mass.sum().item()
        sum_top10_mass += top10_mass.sum().item()

        topk_sinkhorn, _ = torch.topk(visible_assignments_norm, k=max_k, dim=-1)
        top1_prob_sinkhorn = topk_sinkhorn[:, 0]
        top2_prob_sinkhorn = topk_sinkhorn[:, :min(2, max_k)].sum(dim=-1)
        top3_mass_sinkhorn = topk_sinkhorn[:, :min(3, max_k)].sum(dim=-1)
        top5_mass_sinkhorn = topk_sinkhorn[:, :min(5, max_k)].sum(dim=-1)
        top10_mass_sinkhorn = topk_sinkhorn.sum(dim=-1)

        sum_top1_mass_sinkhorn += top1_prob_sinkhorn.sum().item()
        sum_top2_mass_sinkhorn += top2_prob_sinkhorn.sum().item()
        sum_top3_mass_sinkhorn += top3_mass_sinkhorn.sum().item()
        sum_top5_mass_sinkhorn += top5_mass_sinkhorn.sum().item()
        sum_top10_mass_sinkhorn += top10_mass_sinkhorn.sum().item()

        if hungarian_assignments is not None:
            h_pred = hungarian_assignments[b, visible_indices]
            worm_exact_accs_hungarian.append((h_pred == visible_targets).sum().item() / num_nodes)

        if dustbin_weights is not None:
            visible_dustbin = dustbin_weights[b, visible_indices]
            row_sum_total = row_sum_canon.squeeze(-1) + visible_dustbin
            dustbin_share = visible_dustbin / row_sum_total.clamp_min(1e-12)
            sum_dustbin_share += dustbin_share.sum().item()

        if dustbin_row_weights is not None:
            canonical_mass = visible_assignments.sum(dim=0)
            dustbin_mass = dustbin_row_weights[b]
            total_col_mass = (canonical_mass + dustbin_mass).clamp_min(1e-12)
            dustbin_row_share = dustbin_mass / total_col_mass
            sum_dustbin_row_share += dustbin_row_share.sum().item()
            total_canonical_slots += dustbin_row_share.numel()

        correct_prob = visible_probs.gather(1, visible_targets.unsqueeze(1)).squeeze(1)
        correct_prob_sinkhorn = visible_assignments_norm.gather(1, visible_targets.unsqueeze(1)).squeeze(1)
        mispredict_mask = predicted_ids != visible_targets

        if mispredict_mask.any():
            gap = top1_prob[mispredict_mask] - correct_prob[mispredict_mask]
            sum_gap_top1_correct += gap.sum().item()
            ranks = (visible_probs > correct_prob.unsqueeze(1)).sum(dim=1) + 1
            sum_rank_top1_correct += ranks[mispredict_mask].sum().item()

            gap_sinkhorn = top1_prob_sinkhorn[mispredict_mask] - correct_prob_sinkhorn[mispredict_mask]
            sum_gap_top1_correct_sinkhorn += gap_sinkhorn.sum().item()
            ranks_sinkhorn = (visible_assignments_norm > correct_prob_sinkhorn.unsqueeze(1)).sum(dim=1) + 1
            sum_rank_top1_correct_sinkhorn += ranks_sinkhorn[mispredict_mask].sum().item()

            mispredicted_nodes += mispredict_mask.sum().item()

        total_nodes_processed += num_nodes

    sum_exact = sum(worm_exact_accs)
    sum_exact_sq = sum(acc ** 2 for acc in worm_exact_accs)
    count = len(worm_exact_accs)

    sum_top3 = sum(worm_top3_accs)
    sum_top3_sq = sum(acc ** 2 for acc in worm_top3_accs)

    sum_top5 = sum(worm_top5_accs)
    sum_top5_sq = sum(acc ** 2 for acc in worm_top5_accs)

    mean_exact_acc = sum(worm_exact_accs) / len(worm_exact_accs) if worm_exact_accs else 0.0
    mean_top3_acc = sum(worm_top3_accs) / len(worm_top3_accs) if worm_top3_accs else 0.0
    mean_top5_acc = sum(worm_top5_accs) / len(worm_top5_accs) if worm_top5_accs else 0.0

    mean_exact_acc_logits = sum(worm_exact_accs_logits) / len(worm_exact_accs_logits) if worm_exact_accs_logits else 0.0
    mean_top3_acc_logits = sum(worm_top3_accs_logits) / len(worm_top3_accs_logits) if worm_top3_accs_logits else 0.0
    mean_top5_acc_logits = sum(worm_top5_accs_logits) / len(worm_top5_accs_logits) if worm_top5_accs_logits else 0.0

    results = {
        'exact_accuracy': mean_exact_acc,
        'top3_accuracy': mean_top3_acc,
        'top5_accuracy': mean_top5_acc,
        'total_nodes': total_nodes_processed,
        'num_worms': len(worm_exact_accs),
        'node_avg_top1_mass': sum_top1_mass / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top2_mass': sum_top2_mass / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top3_mass': sum_top3_mass / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top5_mass': sum_top5_mass / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top10_mass': sum_top10_mass / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_gap_top1_correct': sum_gap_top1_correct / mispredicted_nodes if mispredicted_nodes else 0.0,
        'node_avg_rank_top1_correct': sum_rank_top1_correct / mispredicted_nodes if mispredicted_nodes else 0.0,
        'node_avg_top1_mass_sinkhorn': sum_top1_mass_sinkhorn / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top2_mass_sinkhorn': sum_top2_mass_sinkhorn / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top3_mass_sinkhorn': sum_top3_mass_sinkhorn / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top5_mass_sinkhorn': sum_top5_mass_sinkhorn / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_top10_mass_sinkhorn': sum_top10_mass_sinkhorn / total_nodes_processed if total_nodes_processed else 0.0,
        'node_avg_gap_top1_correct_sinkhorn': sum_gap_top1_correct_sinkhorn / mispredicted_nodes if mispredicted_nodes else 0.0,
        'node_avg_rank_top1_correct_sinkhorn': sum_rank_top1_correct_sinkhorn / mispredicted_nodes if mispredicted_nodes else 0.0,
        'avg_nodes': total_nodes_processed / max(len(worm_exact_accs), 1),
        'exact_accuracy_sum': sum_exact,
        'exact_accuracy_sum_sq': sum_exact_sq,
        'exact_accuracy_count': count,
        'top3_accuracy_sum': sum_top3,
        'top3_accuracy_sum_sq': sum_top3_sq,
        'top3_accuracy_count': count,
        'top5_accuracy_sum': sum_top5,
        'top5_accuracy_sum_sq': sum_top5_sq,
        'top5_accuracy_count': count,
        'exact_accuracy_logits': mean_exact_acc_logits,
        'top3_accuracy_logits': mean_top3_acc_logits,
        'top5_accuracy_logits': mean_top5_acc_logits,
    }

    if dustbin_weights is not None and total_nodes_processed:
        results['node_avg_dustbin_share'] = sum_dustbin_share / total_nodes_processed

    if hungarian_assignments is not None and worm_exact_accs_hungarian:
        results['exact_accuracy_hungarian'] = sum(worm_exact_accs_hungarian) / len(worm_exact_accs_hungarian)

    if dustbin_row_weights is not None and total_canonical_slots:
        results['canonical_avg_dustbin_row_share'] = (
            sum_dustbin_row_share / total_canonical_slots
        )

    # --- SuperGlue-specific metrics (optional) ---
    if full_assignments is not None:
        sg_metrics = _compute_superglue_metrics(
            full_assignments, targets['visible_mask'], targets['labels']
        )
        results.update(sg_metrics)

    return results

# TODO: To be deleted at some point
 # if self.one_to_one:
        #     # Build (N+1) x (C+1) cost matrix so both unmatched nodes and IDs have slack.
        #     dustbin_col = (self.dustbin_col_score / self.temperature).expand(B, N, 1)
        #     aug_scores = torch.cat([scaled_scores, dustbin_col], dim=-1)

        #     dustbin_row = (self.dustbin_row_score / self.temperature).expand(B, 1, C)
        #     dustbin_corner = (self.dustbin_corner_score / self.temperature).expand(B, 1, 1)
        #     aug_scores = torch.cat([aug_scores, torch.cat([dustbin_row, dustbin_corner], dim=-1)], dim=1)

        #     log_mu = torch.full((B, N + 1), float('-inf'), device=device)
        #     log_nu = torch.full((B, C + 1), float('-inf'), device=device)

        #     for b in range(B):
        #         visible = visible_mask[b]
        #         visible_idx = torch.where(visible)[0]
        #         num_visible = int(visible_idx.numel())
        #         K = max(num_visible, C)

        #         if num_visible > 0:
        #             log_mu[b, visible_idx] = math.log(1.0 / K)

        #         leftover_rows = K - num_visible
        #         log_mu[b, N] = math.log(leftover_rows / K) if leftover_rows > 0 else float('-inf')

        #         log_nu[b, :C] = math.log(1.0 / K)
        #         leftover_cols = K - C
        #         log_nu[b, C] = math.log(leftover_cols / K) if leftover_cols > 0 else float('-inf')

        #     full_assignments = log_sinkhorn_iterations(aug_scores, log_mu, log_nu, self.num_iterations)

        #     # Ks = []
        #     # for b in range(B):
        #     #     visible = visible_mask[b]
        #     #     num_visible = int(visible.sum())
        #     #     K = max(num_visible, C)
        #     #     Ks.append(K)

        #     # Ks = torch.tensor(Ks, device=device, dtype=full_assignments.dtype)  # [B]
        #     # full_assignments = full_assignments * Ks.view(B, 1, 1)

        #     soft_assignments = full_assignments[:, :N, :C]
        #     dustbin_assignments = full_assignments[:, :N, C:C + 1]
        #     dustbin_row_assignments = full_assignments[:, N:N + 1, :C]


    # Each loss term has a weight of 1. Maybe, they can be averaged together. Not sure what is the right way.
    # def _superglue_loss(self, full_P, visible_mask, labels):
    #     """
    #     full_P: [B, N+1, C+1]   (last row/col are dustbins)
    #     visible_mask: [B, N]    (True = real node, False = padding)
    #     labels: [B, N]          (0..C-1 = matched canonical ID, -1 = unmatched)
    #     """
    #     B, N_plus, C_plus = full_P.shape
    #     N = N_plus - 1           # real rows
    #     C = C_plus - 1           # real columns
    #     dustbin_row = N          # last row index
    #     dustbin_col = C          # last col index
    #     eps = 1e-12

    #     device = full_P.device
    #     dtype = full_P.dtype

    #     # NEW scaling
    #     scale = torch.ones(B, 1, 1, device=device, dtype=dtype)
    #     for b in range(B):
    #         m = int(visible_mask[b].sum().item())
    #         if m > 0:
    #             scale[b, 0, 0] = m + C
    #         else:
    #             scale[b, 0, 0] = 1.0

    #     P_scaled = full_P * scale
    #     log_P = (P_scaled.clamp_min(eps)).log()

    #     match_terms = []
    #     unmatched_row_terms = []
    #     unmatched_col_terms = []

    #     for b in range(B):
    #         vis = visible_mask[b].bool()        # [N]
    #         vis_idx = vis.nonzero(as_tuple=False).squeeze(-1)  # indices of visible rows

    #         if vis_idx.numel() == 0:
    #             continue

    #         labels_b = labels[b, vis_idx]       # labels only for visible rows

    #         # --- Matched rows: label >= 0  → (i, label) in M
    #         matched_mask = labels_b >= 0
    #         if matched_mask.any():
    #             matched_rows = vis_idx[matched_mask]
    #             matched_cols = labels_b[matched_mask]          # in [0, C-1]

    #             match_terms.append(-log_P[b, matched_rows, matched_cols].mean())

    #         # --- Unmatched visible rows: label == -1 → should go to dustbin column
    #         unmatched_mask = labels_b < 0
    #         if unmatched_mask.any():
    #             unmatched_rows = vis_idx[unmatched_mask]
    #             unmatched_row_terms.append(-log_P[b, unmatched_rows, dustbin_col].mean())

    #         # --- Unused canonical IDs: columns in J → should go to dustbin row
    #         used_mask = torch.zeros(C, device=device, dtype=torch.bool)
    #         if matched_mask.any():
    #             used_mask.scatter_(0, labels_b[matched_mask], True)

    #         unused_cols = (~used_mask).nonzero(as_tuple=False).squeeze(-1)
    #         if unused_cols.numel() > 0:
    #             unmatched_col_terms.append(-log_P[b, dustbin_row, unused_cols].mean())

    #     terms = []
    #     if match_terms:
    #         terms.append(torch.stack(match_terms).mean())
    #     if unmatched_row_terms:
    #         terms.append(torch.stack(unmatched_row_terms).mean())
    #     if unmatched_col_terms:
    #         terms.append(torch.stack(unmatched_col_terms).mean())

    #     if terms:
    #         return torch.stack(terms).sum()
    #     return torch.tensor(0.0, device=full_P.device)
