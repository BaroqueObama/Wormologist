"""
Muon Optimizer - MomentUm Orthogonalized by Newton-schulz

This is from: https://github.com/MoonshotAI/Moonlight
Muon is designed for 2D parameter matrices in neural networks, using orthogonalized momentum updates.
"""

import math
import torch
from typing import List, Optional, Callable


@torch.compile
def zeropower_via_newtonschulz5(G, steps, use_bfloat16=True):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Use bfloat16 if supported and requested, otherwise use float16
    if use_bfloat16 and G.device.type == 'cuda' and torch.cuda.is_bf16_supported():
        X = G.bfloat16()
    else:
        X = G.float16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW (can be same as lr or different).
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        muon_params=None,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_lr=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0.01,
        use_bfloat16=True,
    ):
        # Use adamw_lr if specified, otherwise use lr
        if adamw_lr is None:
            adamw_lr = lr

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
            use_bfloat16=use_bfloat16,
        )

        params = list(muon_params) if muon_params else []
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in (muon_params if muon_params else []):
            # Only use Muon for 2D parameters
            if p.ndim == 2:
                self.state[p]["use_muon"] = True
            else:
                # Non-2D parameters in muon_params get AdamW
                self.state[p]["use_muon"] = False
                
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        """
        Adjust the learning rate based on the size of the parameter matrix
        as described in the paper. This helps maintain stable training across
        different parameter sizes.
        """
        A, B = param_shape[:2]
        # We adjust the learning rate based on the size of the parameter matrix
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            lr = group["lr"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim != 2:
                    # Skip non-2D gradients
                    continue
                    
                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                    
                # Orthogonalize the momentum
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"], use_bfloat16=group.get("use_bfloat16", True))

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay directly to weights (not to gradient)
                if group.get("adamw_wd", 0) > 0:
                    p.data.mul_(1 - lr * group["adamw_wd"])

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p].get("use_muon", False)]
            lr = group.get("adamw_lr", group["lr"])
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                    
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
                # Weight decay
                p.data.mul_(1 - lr * weight_decay)
                # Apply update
                p.data.add_(g, alpha=-lr / scale)

        return loss
