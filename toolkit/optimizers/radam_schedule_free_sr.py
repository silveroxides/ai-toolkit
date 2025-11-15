"""
Stochastic Rounding version of Schedule-Free RAdam (RAdamScheduleFreeSR)
=======================================================================

This implementation is adapted from Facebook AI Research's **schedule-free** repository:
https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/radam_schedulefree.py


"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from typing_extensions import TypeAlias

try:
    from torch.optim.optimizer import ParamsT  # PyTorch ≥ 2.2 ships this alias
except ImportError:  # pragma: no cover
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

from toolkit.optimizers.optimizer_utils import (
    stochastic_grad_accummulation,
    copy_stochastic,
)


class RAdamScheduleFreeSR(torch.optim.Optimizer):
    r"""Schedule‑Free RAdam (stochastic‑rounding edition).

    A warm‑up‑free variant of *Rectified Adam* [Liu et al., 2020] with the adaptive
    re‑parameterisation trick of *Schedule‑Free Optimisation* [Chen et al., 2023].
    This version can write parameters with **stochastic rounding** so that training
    with reduced precision is unbiased.

    The optimizer **must** be toggled between training and evaluation phases by
    calling :py:meth:`train` and :py:meth:`eval` respectively, as in the original
    Schedule‑Free implementation.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 2.5e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: Optional[bool] = False,
        silent_sgd_phase: bool = True,
        stochastic_rounding: bool = True,
    ) -> None:
        if foreach:
            raise ValueError(
                "foreach kernels are disabled in RAdamScheduleFreeSR because they "
                "do not expose hooks for stochastic rounding. Pass foreach=False "
                "(default) or remove the argument."
            )
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,  # global step counter per param‑group
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            foreach=False,
            silent_sgd_phase=silent_sgd_phase,
        )
        super().__init__(params, defaults)

        # Determine if stochastic rounding for updates should be performed
        self.perform_stochastic_rounding_on_update = False
        if stochastic_rounding:
            can_do_sr_for_updates = True
            for group in self.param_groups:
                for param in group["params"]:
                    if param.dtype == torch.float32:
                        can_do_sr_for_updates = False
                        print(
                            "Warning: RAdamScheduleFreeSR stochastic_rounding=True was requested, "
                            "but float32 parameters detected. Disabling stochastic rounding for "
                            "parameter updates as .float() would be a no-op (not a copy). "
                            "Standard float32 operations will be used for updates."
                        )
                        break
                if not can_do_sr_for_updates:
                    break
            if can_do_sr_for_updates:
                self.perform_stochastic_rounding_on_update = True

        # Setup stochastic grad accumulation hooks
        self.is_stochastic_rounding_accumulation = False
        if (
            self.perform_stochastic_rounding_on_update
        ):  # Only relevant if SR updates are active
            for group in self.param_groups:
                for param in group["params"]:
                    if param.requires_grad and param.dtype != torch.float32:
                        # This hook is for low-precision gradients being accumulated into higher precision
                        self.is_stochastic_rounding_accumulation = True
                        param.register_post_accumulate_grad_hook(
                            stochastic_grad_accummulation
                        )
                        # Found one, no need to set the flag multiple times, but continue registering hooks

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # Copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def eval(self) -> None:
        """Switch the optimiser to *evaluation* mode (a.k.a. **x‑space**)."""
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            inv_beta1 = 1.0 / beta1
            weight = 1.0 - inv_beta1

            for p in group["params"]:
                state = self.state[p]
                z = state.get("z")
                if z is None:
                    continue

                if self.perform_stochastic_rounding_on_update:
                    # Stochastic rounding path (p.dtype is not float32)
                    p_fp32 = p.float()
                    z_fp32 = z.float()
                    # Equivalent to: result_fp32 = p_fp32 + weight * (z_fp32 - p_fp32)
                    result_fp32 = p_fp32.clone()  # Make a mutable copy
                    result_fp32.add_(z_fp32 - p_fp32, alpha=weight)
                    copy_stochastic(p, result_fp32)
                else:
                    # Non-stochastic rounding path
                    if p.dtype == torch.float32:
                        # Optimal path for float32 master weights: in-place lerp
                        p.lerp_(z, weight)
                    else:
                        # Path for non-float32 master weights (e.g. bf16) with SR disabled
                        # Calculate in fp32 and copy back (standard cast)
                        p_fp32 = p.float()
                        z_fp32 = z.float()  # z has same dtype as p
                        result_fp32 = torch.lerp(p_fp32, z_fp32, weight)
                        p.copy_(
                            result_fp32
                        )  # Standard copy, handles potential downcast
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Switch the optimiser to *training* mode (a.k.a. **y‑space**)."""
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            weight = 1.0 - beta1

            for p in group["params"]:
                state = self.state[p]
                z = state.get("z")
                if z is None:
                    continue

                if self.perform_stochastic_rounding_on_update:
                    # Stochastic rounding path (p.dtype is not float32)
                    p_fp32 = p.float()
                    z_fp32 = z.float()
                    result_fp32: torch.Tensor
                    result_fp32 = p_fp32.clone()
                    result_fp32.add_(z_fp32 - p_fp32, alpha=weight)
                    copy_stochastic(p, result_fp32)
                else:
                    # Non-stochastic rounding path
                    if p.dtype == torch.float32:
                        # Optimal path for float32 master weights: in-place lerp
                        p.lerp_(z, weight)
                    else:
                        # Path for non-float32 master weights (e.g. bf16) with SR disabled
                        p_fp32 = p.float()
                        z_fp32 = z.float()
                        result_fp32 = torch.lerp(p_fp32, z_fp32, weight)
                        p.copy_(result_fp32)
            group["train_mode"] = True

    # ------------------------------------------------------------------
    # Main optimisation step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform **one** optimisation step."""
        self.step_hook()

        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "Optimizer is in eval mode. Call optimizer.train() before "
                "back‑propagating and optimizer.eval() before validation/check‑pointing."
            )

        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay = group["weight_decay"]
            silent_sgd_phase = group["silent_sgd_phase"]
            r = group["r"]
            weight_lr_power = group["weight_lr_power"]

            k = group["k"]
            step_num = k + 1

            beta2_t = beta2**step_num
            bias_correction2 = 1.0 - beta2_t
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            rho_t = rho_inf - 2.0 * step_num * beta2_t / bias_correction2

            rect: float
            if rho_t > 4.0:
                rect = math.sqrt(
                    (rho_t - 4.0)
                    * (rho_t - 2.0)
                    * rho_inf
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                )
            else:
                rect = float(not silent_sgd_phase)

            lr_scheduled = group["lr"] * rect
            group["scheduled_lr"] = lr_scheduled
            group["lr_max"] = lr_max = max(lr_scheduled, group["lr_max"])

            weight = (step_num**r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] + weight
            group["weight_sum"] = weight_sum
            ckp1 = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr = lr_scheduled * (beta1 * (1.0 - ckp1) - 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad  # Original grad, could be bf16, fp16, etc.
                state = self.state[p]

                # --- Lazy state initialisation (match p's dtype)
                if len(state) == 0:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                z = state["z"]  # Matches p's dtype
                exp_avg_sq = state["exp_avg_sq"]  # Matches p's dtype

                # --- Second‑moment update (RMS of gradients)
                # Performed in exp_avg_sq's dtype (e.g., bf16)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # --- Prepare for fp32 calculations
                # grad_fp32 will be a new tensor if grad is not fp32
                grad_fp32 = grad.float() if grad.dtype != torch.float32 else grad

                grad_normalized_fp32: torch.Tensor
                if rho_t > 4.0:
                    # Denominator calculation in fp32 for stability
                    # exp_avg_sq_fp32 will be a new tensor if exp_avg_sq is not fp32
                    exp_avg_sq_fp32 = (
                        exp_avg_sq.float()
                        if exp_avg_sq.dtype != torch.float32
                        else exp_avg_sq
                    )

                    # Use a clone for in-place ops if exp_avg_sq_fp32 is not a new tensor (i.e. was already fp32)
                    # to avoid modifying state['exp_avg_sq'] if it's already fp32.
                    # However, since exp_avg_sq_fp32 is used as a temporary for denom, it's fine.
                    denom_fp32 = (
                        exp_avg_sq_fp32.div(bias_correction2).sqrt_().add_(eps)
                    )  # In-place on exp_avg_sq_fp32 or its clone
                    grad_normalized_fp32 = grad_fp32 / denom_fp32
                else:
                    grad_normalized_fp32 = grad_fp32  # Already float32

                # --- Weight decay (applied in y‑space, using fp32 representations)
                if decay != 0.0:
                    # p_fp32_for_decay will be a new tensor if p is not fp32
                    p_fp32_for_decay = p.float() if p.dtype != torch.float32 else p
                    grad_normalized_fp32.add_(p_fp32_for_decay, alpha=decay)

                # ----------------------------------------------------------
                # y - update (parameter p)
                # ----------------------------------------------------------
                if self.perform_stochastic_rounding_on_update:
                    # SR path: p.dtype is not float32
                    # Calculate update in fp32, then copy_stochastic
                    p_fp32 = p.float()  # New tensor
                    z_fp32 = z.float()  # New tensor (z has same dtype as p)

                    # temp_p_fp32 = (1-ckp1)*p_fp32 + ckp1*z_fp32
                    temp_p_fp32 = torch.lerp(p_fp32, z_fp32, ckp1)
                    temp_p_fp32.add_(grad_normalized_fp32, alpha=adaptive_y_lr)
                    copy_stochastic(p, temp_p_fp32)
                    del temp_p_fp32, p_fp32, z_fp32
                else:
                    # Non-SR path
                    if p.dtype == torch.float32:
                        # Optimal path for float32 master weights: in-place ops
                        # p = (1-ckp1)*p + ckp1*z (lerp part)
                        p.mul_(1.0 - ckp1)
                        p.add_(z, alpha=ckp1)  # z is also float32
                        # p = p + adaptive_y_lr * grad_normalized_fp32
                        p.add_(grad_normalized_fp32, alpha=adaptive_y_lr)
                    else:
                        # Path for non-float32 master weights (e.g. bf16) with SR disabled
                        # Calculate in fp32, then copy back (standard cast)
                        p_fp32 = p.float()  # New tensor
                        z_fp32 = z.float()  # New tensor

                        temp_p_fp32 = torch.lerp(p_fp32, z_fp32, ckp1)
                        temp_p_fp32.add_(grad_normalized_fp32, alpha=adaptive_y_lr)
                        p.copy_(
                            temp_p_fp32
                        )  # Standard copy, handles potential downcast
                        del temp_p_fp32, p_fp32, z_fp32

                # ----------------------------------------------------------
                # z - update (SGD‑style)
                # ----------------------------------------------------------
                if self.perform_stochastic_rounding_on_update:
                    # SR path: z.dtype is not float32
                    # Calculate update in fp32, then copy_stochastic
                    z_fp32 = z.float()  # New tensor

                    # temp_z_fp32 = z_fp32 - lr_scheduled * grad_normalized_fp32
                    # .sub creates a new tensor by default
                    temp_z_fp32 = z_fp32.sub(grad_normalized_fp32, alpha=lr_scheduled)
                    copy_stochastic(z, temp_z_fp32)
                    del temp_z_fp32, z_fp32
                else:
                    # Non-SR path
                    if z.dtype == torch.float32:  # Same as p.dtype
                        # Optimal path for float32 master weights: in-place op
                        z.sub_(grad_normalized_fp32, alpha=lr_scheduled)
                    else:
                        # Path for non-float32 master weights (e.g. bf16) with SR disabled
                        # Calculate in fp32, then copy back (standard cast)
                        z_fp32 = z.float()  # New tensor
                        temp_z_fp32 = z_fp32.sub(
                            grad_normalized_fp32, alpha=lr_scheduled
                        )
                        z.copy_(
                            temp_z_fp32
                        )  # Standard copy, handles potential downcast
                        del temp_z_fp32, z_fp32

            group["k"] = step_num
        return loss
