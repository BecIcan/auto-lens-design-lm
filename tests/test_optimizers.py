from types import SimpleNamespace

import torch

from eisoptx.optimization.optimizers import LMOptimizer


DTYPE = torch.float64


class LinearConstraintClosure:
    def __init__(self, parameter):
        self.parameter = parameter
        self.loss_wrapper = SimpleNamespace(
            residual_slices={"objective": slice(0, 2), "constraint": slice(2, None)}
        )

    def quantities(self):
        x, y = self.parameter
        residual = torch.stack((x - 1.0, y - 3.0, x + y - 2.0))
        jacobian = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=DTYPE
        )
        mask = torch.tensor([False, False, True])
        scalar = 0.5 * residual[~mask].square().sum()
        return residual, scalar, jacobian, mask

    def get_least_squares_quantities(self):
        return self.quantities()

    def evaluate_least_squares_loss(self):
        return self.quantities()[1]

    def evaluate_constraint_norms(self):
        residual, _, _, mask = self.quantities()
        constraints = residual[mask]
        return float(constraints.norm()), float(constraints.abs().max())


def test_lm_null_space_step_satisfies_linear_constraint_and_reduces_loss():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 0.0], dtype=DTYPE))
    optimizer = LMOptimizer([parameter], lm_parameter=1e-6, tolerance=1.0)
    closure = LinearConstraintClosure(parameter)
    wrapped_closure = SimpleNamespace(args=(None, closure))
    initial_loss = closure.evaluate_least_squares_loss()

    optimizer.step(wrapped_closure)

    torch.testing.assert_close(parameter.sum(), torch.tensor(2.0, dtype=DTYPE))
    assert closure.evaluate_least_squares_loss() < initial_loss
    assert optimizer.logs["constraint_rank"] == 1
    assert optimizer.logs["reduced_rank"] == 1
    assert optimizer.logs["step_accepted"] == 1
    assert optimizer.logs["candidate_constraint_norm"] < 1e-12
    assert 0.0 <= optimizer.logs["projected_gradient_ratio_objective"] <= 1.0


class NonlinearConstraintClosure:
    def __init__(self, parameter, residual_target=0.0):
        self.parameter = parameter
        self.residual_target = residual_target

    def quantities(self):
        (x,) = self.parameter
        residual = torch.stack((x - self.residual_target, x.square() - 1.0))
        jacobian = torch.stack((torch.ones_like(x), 2.0 * x)).reshape(2, 1)
        mask = torch.tensor([False, True])
        scalar = 0.5 * residual[~mask].square().sum()
        return residual, scalar, jacobian, mask

    def get_least_squares_quantities(self):
        return self.quantities()

    def evaluate_least_squares_loss(self):
        return self.quantities()[1]

    def evaluate_constraint_norms(self):
        residual, _, _, mask = self.quantities()
        constraints = residual[mask]
        return float(constraints.norm()), float(constraints.abs().max())


def test_lm_rejects_step_that_worsens_nonlinear_constraint():
    parameter = torch.nn.Parameter(torch.tensor([0.1], dtype=DTYPE))
    optimizer = LMOptimizer(
        [parameter],
        lm_parameter=1e-6,
        tolerance=100.0,
        constraint_increase_factor=1.0,
        constraint_reduction_factor=0.1,
    )
    closure = NonlinearConstraintClosure(parameter)
    wrapped_closure = SimpleNamespace(args=(None, closure))
    before = parameter.detach().clone()

    optimizer.step(wrapped_closure)

    torch.testing.assert_close(parameter, before)
    assert optimizer.logs["constraint_accepted"] == 0
    assert optimizer.logs["step_accepted"] == 0


def test_lm_filter_accepts_sufficient_nonlinear_feasibility_progress():
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=DTYPE))
    optimizer = LMOptimizer(
        [parameter],
        lm_parameter=1e-6,
        tolerance=1.0,
        constraint_reduction_factor=0.9,
    )
    closure = NonlinearConstraintClosure(parameter, residual_target=10.0)
    wrapped_closure = SimpleNamespace(args=(None, closure))

    optimizer.step(wrapped_closure)

    assert optimizer.logs["objective_accepted"] == 0
    assert optimizer.logs["feasibility_progress"] == 1
    assert optimizer.logs["step_accepted"] == 1
    assert abs(float(parameter.detach()) - 1.25) < 1e-8


class RankDeficientConstraintClosure(LinearConstraintClosure):
    def quantities(self):
        x, y = self.parameter
        residual = torch.stack(
            (x - 1.0, y - 3.0, x + y - 2.0, 2.0 * x + 2.0 * y - 4.0)
        )
        jacobian = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]], dtype=DTYPE
        )
        mask = torch.tensor([False, False, True, True])
        scalar = 0.5 * residual[~mask].square().sum()
        return residual, scalar, jacobian, mask


def test_lm_handles_rank_deficient_constraints():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 0.0], dtype=DTYPE))
    optimizer = LMOptimizer([parameter], lm_parameter=1e-6, tolerance=1.0)
    closure = RankDeficientConstraintClosure(parameter)
    wrapped_closure = SimpleNamespace(args=(None, closure))

    optimizer.step(wrapped_closure)

    torch.testing.assert_close(parameter.sum(), torch.tensor(2.0, dtype=DTYPE))
    assert optimizer.logs["constraint_rank"] == 1
    assert optimizer.logs["constraint_sigma_min"] > 0
    assert optimizer.logs["step_accepted"] == 1
