import torch

from eisoptx.modeling import ray_analysis as ra


def test_rms_ignores_invalid_rays_without_rewarding_failures():
    y = torch.tensor([0.0, 2.0, 1000.0], dtype=torch.float64)
    x = torch.zeros_like(y)
    valid = torch.tensor([True, True, False])

    rms = ra.compute_rms_spot_size(x, y, valid, reduce_dims=(0,), eps=0.0)

    assert torch.allclose(rms, torch.tensor(1.0, dtype=torch.float64))


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
