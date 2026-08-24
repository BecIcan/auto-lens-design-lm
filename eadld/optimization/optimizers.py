import warnings
import math
from typing import Iterable

import torch
import torch.optim
import numpy as np


class LMOptimizer(torch.optim.Optimizer):
    """带等式约束的 Levenberg–Marquardt 优化器。

    无约束步求解 `min ||r + J dx||² + lambda ||D dx||²`。存在约束
    `A dx = -c` 时，先求特解 `dx_p`，再令 `dx = dx_p + N y`，
    在约束零空间 `N` 内完成 Gauss–Newton 二阶近似。
    """

    def __init__(
        self,
        params: Iterable,
        lm_parameter: float = 1.0,
        damped_term_min: float = 1e-4,
        tolerance: float = 2.0,
        lam_increase_factor: float = 2.0,
        lam_decrease_factor: float = 2.0,
        lam_eps: float = 1e-6,
        beta: float = 0.99,
        constraint_tolerance: float = 1e-9,
        constraint_increase_factor: float = 1.05,
        constraint_reduction_factor: float = 0.9,
    ):
        """Constructor.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lm_parameter: Initial value of the Levenberg-Marquardt parameter.
            damped_term_min: Minimum value of the damped term (to avoid large jumps when gradients are small).
            tolerance: If the scalar loss increases by more than this factor, the step is rejected.
            lam_increase_factor: Factor by which to increase the LM parameter when the loss increases.
            lam_decrease_factor: Factor by which to decrease the LM parameter when the loss decreases.
            lam_eps: LM parameter is bounded within [lam_eps, lam_eps ** -1].
            beta: Multiplier for running mean of damping matrix.
            constraint_tolerance: Absolute L2 feasibility tolerance for constraints.
            constraint_increase_factor: Maximum accepted relative constraint increase.
            constraint_reduction_factor: Sufficient feasibility reduction for filter acceptance.
        """
        defaults = {
            "lm_parameter": lm_parameter,
            "damped_term_min": damped_term_min,
            "tolerance": tolerance,
            "lam_increase_factor": lam_increase_factor,
            "lam_decrease_factor": lam_decrease_factor,
            "lam_eps": lam_eps,
            "beta": beta,
            "constraint_tolerance": constraint_tolerance,
            "constraint_increase_factor": constraint_increase_factor,
            "constraint_reduction_factor": constraint_reduction_factor,
            "damping_terms": None,
        }
        super(LMOptimizer, self).__init__(params, defaults)
        self.logs = {}

    @torch.no_grad()
    def step(self, closure):
        """Perform a single LM optimization step, evaluate the loss and conditionally update the parameters.

        Args:
            closure (callable): An object with a function to return the necessary data for LM optimization.
        """
        closure = closure.args[-1]  # Remove wrapper
        logs = {}

        assert len(self.param_groups) == 1, (
            "LMOptimizer does not support per-parameter options (parameter groups)."
        )
        group = self.param_groups[0]

        lam = group["lm_parameter"]
        min_damp = group["damped_term_min"]
        logs["lm_parameter"] = lam

        with torch.enable_grad():
            loss, scalar_loss, jacobian, constraint_mask = (
                closure.get_least_squares_quantities()
            )

        residual_jacobian = jacobian[~constraint_mask]
        constraint_jacobian = jacobian[constraint_mask]
        residuals = loss[~constraint_mask]
        constraints = loss[constraint_mask]
        logs["n_residuals"] = float(residuals.numel())
        initial_constraint_norm = float(constraints.norm())
        initial_constraint_max = (
            float(constraints.abs().max()) if constraints.numel() else 0.0
        )
        logs["constraint_norm"] = initial_constraint_norm
        logs["constraint_max"] = initial_constraint_max

        # D 取 Jacobian 列范数，避免不同物理量尺度主导阻尼。
        damping_terms = residual_jacobian.norm(
            dim=0
        )  # Compute sqrt of diagonal elements of pseudo-Hessian
        if group["damping_terms"] is None:
            damping_terms = damping_terms.clamp(min=min_damp)
        else:
            beta = group["beta"]
            damping_terms = (
                beta * torch.max(damping_terms, group["damping_terms"])
                + (1 - beta) * damping_terms
            )

        if (
            constraints.numel() > 0
        ):  # If there are constraints, solve constrained optimization problem
            # 先解 A dx_p = -c，再在零空间 N 中优化；不显式形成 J^T J，
            # 可避免条件数平方，并让论文中的分支约束直接对应可行步。
            a = constraint_jacobian.cpu()
            c = constraints.cpu()
            constraint_solve = torch.linalg.lstsq(a, -c, driver="gelsd")
            particular_step = constraint_solve.solution
            logs["particular_step_norm"] = float(particular_step.norm())
            constraint_rank = int(constraint_solve.rank.item())
            logs["constraint_rank"] = float(constraint_rank)

            svd = torch.linalg.svd(a, full_matrices=True)
            vh = svd.Vh
            if constraint_rank:
                smallest = float(svd.S[constraint_rank - 1])
                logs["constraint_sigma_min"] = smallest
                logs["constraint_condition"] = float(svd.S[0]) / max(
                    smallest, torch.finfo(a.dtype).tiny
                )
            null_space = vh[constraint_rank:].T
            loss_wrapper = getattr(closure, "loss_wrapper", None)
            residual_slices = getattr(loss_wrapper, "residual_slices", {})
            for name, residual_slice in residual_slices.items():
                if constraint_mask[residual_slice].any():
                    continue
                term_residuals = loss[residual_slice].cpu()
                term_jacobian = jacobian[residual_slice].cpu()
                gradient = term_jacobian.T @ term_residuals
                if null_space.shape[1] == 0:
                    projected_gradient = torch.zeros_like(gradient)
                else:
                    projected_gradient = null_space @ (null_space.T @ gradient)
                gradient_norm = float(gradient.norm())
                projected_norm = float(projected_gradient.norm())
                key = name.replace("/", "_")
                logs[f"gradient_norm_{key}"] = gradient_norm
                logs[f"projected_gradient_norm_{key}"] = projected_norm
                logs[f"projected_gradient_ratio_{key}"] = projected_norm / max(
                    gradient_norm, torch.finfo(gradient.dtype).tiny
                )
            if null_space.shape[1] == 0:
                step = particular_step
                logs["null_space_step_norm"] = 0.0
                logs["reduced_rank"] = 0.0
            else:
                j = residual_jacobian.cpu()
                residuals_cpu = residuals.cpu()
                damp = damping_terms.diag_embed().cpu()
                # min_y ||r + J(dx_p + N y)||² + lambda||D(dx_p + N y)||²
                reduced_matrix = torch.cat(
                    (
                        j @ null_space,
                        np.sqrt(lam) * damp @ null_space,
                    ),
                    dim=0,
                )
                reduced_rhs = -torch.cat(
                    (
                        residuals_cpu + j @ particular_step,
                        np.sqrt(lam) * damp @ particular_step,
                    )
                )
                reduced_solve = torch.linalg.lstsq(
                    reduced_matrix, reduced_rhs, driver="gelsd"
                )
                null_space_step = null_space @ reduced_solve.solution
                step = particular_step + null_space_step
                logs["null_space_step_norm"] = float(null_space_step.norm())
                logs["reduced_rank"] = float(reduced_solve.rank.item())
            step = step.to(jacobian.device)
        else:  # 无约束时直接解增广最小二乘，不显式构造伪 Hessian。
            matrix = torch.cat(
                (residual_jacobian, np.sqrt(lam) * damping_terms.diag_embed()), dim=0
            )
            bb = torch.cat((residuals, torch.zeros_like(damping_terms)), dim=0)
            # Least-squares solver on CPU is more reliable
            try:
                lstsq = torch.linalg.lstsq(matrix.cpu(), -bb.cpu(), driver="gelsd")
                step = lstsq[0].to(matrix.device)
                logs["rank"] = float(lstsq.rank.item())
            except RuntimeError:
                warnings.warn("Least-squares solver failed; step ignored")
                step = None

        # Update parameters
        if step is not None:
            logs["step_norm"] = float(step.norm())
            trainable_params = [v for v in group["params"] if v.requires_grad]
            assert len(trainable_params) == 1, (
                "LMOptimizer does not support multiple parameter tensors"
            )
            p = trainable_params[0]
            p_copy = p.data.clone()
            p.data.add_(step)

            with torch.inference_mode(True):
                updated_loss = closure.evaluate_least_squares_loss()
                updated_constraint_norm, updated_constraint_max = (
                    closure.evaluate_constraint_norms()
                    if hasattr(closure, "evaluate_constraint_norms")
                    else (initial_constraint_norm, initial_constraint_max)
                )
            loss_ratio = (updated_loss / scalar_loss).item()
            logs["loss_ratio"] = loss_ratio
            logs["candidate_constraint_norm"] = updated_constraint_norm
            logs["candidate_constraint_max"] = updated_constraint_max
            constraint_limit = max(
                group["constraint_tolerance"],
                group["constraint_increase_factor"] * initial_constraint_norm,
            )
            constraint_accepted = updated_constraint_norm <= constraint_limit
            logs["constraint_accepted"] = float(constraint_accepted)
            objective_accepted = (
                math.isfinite(loss_ratio)
                and loss_ratio <= group["tolerance"]
                and constraint_accepted
            )
            feasibility_progress = (
                math.isfinite(loss_ratio)
                and initial_constraint_norm > group["constraint_tolerance"]
                and updated_constraint_norm
                <= group["constraint_reduction_factor"] * initial_constraint_norm
            )
            logs["objective_accepted"] = float(objective_accepted)
            logs["feasibility_progress"] = float(feasibility_progress)
            step_accepted = objective_accepted or feasibility_progress
            logs["step_accepted"] = float(step_accepted)
            if not step_accepted:
                # Reject step
                p.data = p_copy
            if step_accepted and loss_ratio <= 1.0:
                lam = lam / group["lam_decrease_factor"]
            else:
                lam = lam * group["lam_increase_factor"]

            # Update optimizer parameters
            # Keep optimizer state checkpoint-safe with PyTorch's
            # ``weights_only=True`` loader (NumPy scalar globals are rejected).
            group["lm_parameter"] = float(
                np.clip(lam, group["lam_eps"], 1 / group["lam_eps"])
            )
            group["damping_terms"] = damping_terms

        # Log
        self.logs = logs

        return scalar_loss
