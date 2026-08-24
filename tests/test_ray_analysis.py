import torch

from eadld.imaging_system import _combine_ray_weights
from eadld.modeling import ray_analysis as ra
from eadld.optimization.residuals import TransverseRayAberrationResiduals


def test_rms_ignores_invalid_rays_without_rewarding_failures():
    y = torch.tensor([0.0, 2.0, 1000.0], dtype=torch.float64)
    x = torch.zeros_like(y)
    valid = torch.tensor([True, True, False])

    rms = ra.compute_rms_spot_size(x, y, valid, reduce_dims=(0,), eps=0.0)

    assert torch.allclose(rms, torch.tensor(1.0, dtype=torch.float64))


def test_rms_ignores_infinite_failed_rays():
    x = torch.tensor([0.0, 1.0, float("inf")], dtype=torch.float64)
    y = torch.zeros_like(x)
    valid = torch.tensor([True, True, False])

    rms = ra.compute_rms_spot_size(x, y, valid, reduce_dims=(0,), eps=0.0)

    assert torch.isfinite(rms)
    assert torch.allclose(rms, torch.tensor(0.5**0.5, dtype=torch.float64))


def test_transverse_residual_penalizes_complete_ray_failure():
    xy = torch.full((2, 1, 4, 3, 1), float("inf"), dtype=torch.float64)
    centroid = torch.zeros((2, 1, 1, 1, 1), dtype=torch.float64)

    residual = TransverseRayAberrationResiduals(weight=1.0)(xy, centroid)

    assert residual.shape == (xy.numel() + 1,)
    assert torch.all(residual[:-1] == 0)
    assert residual[-1].item() == 1.0


def test_weighted_rms_matches_area_equivalent_duplicated_samples():
    x = torch.tensor([0.0, 2.0], dtype=torch.float64)
    y = torch.zeros_like(x)
    valid = torch.ones_like(x, dtype=torch.bool)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)

    weighted = ra.compute_rms_spot_size(
        x, y, valid, reduce_dims=(0,), weights=weights, eps=0.0
    )

    duplicated_x = torch.tensor([0.0, 2.0, 2.0, 2.0], dtype=torch.float64)
    duplicated_y = torch.zeros_like(duplicated_x)
    duplicated_valid = torch.ones_like(duplicated_x, dtype=torch.bool)
    duplicated = ra.compute_rms_spot_size(
        duplicated_x,
        duplicated_y,
        duplicated_valid,
        reduce_dims=(0,),
        eps=0.0,
    )

    assert torch.allclose(weighted, duplicated)


def test_zonal_pupil_weights_work_without_spectral_weights():
    """未设置光谱权重时，环带面积权重仍必须独立生效。"""
    pupil = torch.tensor([1.0, 3.0], dtype=torch.float64)

    assert _combine_ray_weights(None, pupil) is pupil


def test_legacy_replay_can_disable_pupil_quadrature():
    """论文旧轨迹可关闭后来加入的面积求积，其余运行默认保持开启。"""
    from inspect import signature

    from eadld.imaging_system import ImagingSystemModule

    parameter = signature(ImagingSystemModule).parameters[
        "optimization_pupil_quadrature"
    ]
    assert parameter.default is True
