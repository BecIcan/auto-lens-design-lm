import pytest
import torch

from eadld.modeling import optics, ray_tracing
from eadld.optimization.parameterization import LensParameterization


def test_zonal_modifier_attaches_to_existing_surface_events():
    sequence = optics.LensSequence("sRz-")

    assert sequence.n_zonal == 1
    zonal_events = [event for event in sequence.events if "z" in event]
    assert [event["type"] for event in zonal_events] == ["p", "r"]
    assert {event["z"] for event in zonal_events} == {0}

    with pytest.raises(AssertionError, match="both aspherical and zonal"):
        optics.LensSequence("sazR-")
    with pytest.raises(AssertionError, match="must modify refractive interfaces"):
        optics.LensSequence("sz-")


def test_zonal_profile_uses_inner_boundary_and_ignores_padding():
    rho = torch.tensor([[0.25], [1.0], [1.0002], [4.0], [4.1]])
    curvature = torch.tensor([0.0])
    zones = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 1.0],
                [2.0, 0.0, 10.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    sag, inside = ray_tracing.evaluate_zonal_profile(rho, curvature, zones)

    torch.testing.assert_close(sag[:, 0], torch.tensor([0.25, 1.0, 12.0004, 18.0, 0.0]))
    assert inside[:, 0].tolist() == [True, True, True, True, False]


def test_zonal_newton_marching_reaches_offset_facet():
    ray_position = torch.tensor([[[0.5]], [[0.0]], [[-5.0]]])
    ray_direction = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    curvature = torch.tensor([0.0])
    zones = torch.tensor([[[0.0, 0.0, 2.0, 1.0]]])

    distance, miss = ray_tracing.find_marching_distance_zonal(
        ray_position, ray_direction, curvature, zones
    )

    torch.testing.assert_close(distance, torch.tensor([[7.0]]))
    assert not miss.any()


def test_zonal_marching_keeps_fixed_zone_coefficients_differentiable():
    ray_position = torch.tensor([[[0.5]], [[0.0]], [[-5.0]]])
    ray_direction = torch.tensor([[[0.0]], [[0.0]], [[1.0]]])
    curvature = torch.tensor([0.0])
    zones = torch.tensor([[[0.1, 0.01, 2.0, 1.0]]], requires_grad=True)

    distance, miss = ray_tracing.find_marching_distance_zonal(
        ray_position, ray_direction, curvature, zones
    )
    distance.sum().backward()

    assert not miss.any()
    assert torch.isfinite(zones.grad).all()
    assert (zones.grad[0, 0, :3] != 0).all()


# 倾斜光线可能与拼接面相交两次，也可能正好打在台阶侧壁；两种情况都要锁定。
def test_zonal_marching_takes_nearest_facet_and_flags_sidewall_rays():
    curvature = torch.tensor([0.0])
    # Flat facets: zone 0 at z = 0 for r <= 1, zone 1 at z = 0.5 for 1 < r <= 2.
    zones = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.5, 2.0]]])

    def trace(radius_at_vertex, slope):
        direction = torch.tensor([slope, 0.0, 1.0])
        direction = direction / direction.norm()
        start = torch.tensor([radius_at_vertex - 5 * slope, 0.0, -5.0])
        return (
            start[:, None, None],
            direction[:, None, None].expand(3, 1, 1).contiguous(),
        )

    # Crosses zone 0 at z = 0 and zone 1 at z = 0.5; the nearer facet wins.
    r, d = trace(0.999, 0.01)
    distance, miss = ray_tracing.find_marching_distance_zonal(r, d, curvature, zones)
    hit = r + distance * d
    assert not miss.any()
    torch.testing.assert_close(hit[2, 0, 0], torch.tensor(0.0), atol=1e-9, rtol=0)

    # Aimed at the step: outside zone 0, and zone 1's facet lies inside zone 0.
    r, d = trace(1.001, -0.01)
    _, miss = ray_tracing.find_marching_distance_zonal(r, d, curvature, zones)
    assert miss.all()


def test_native_lens_geometry_exposes_zonal_sag_for_layout_plotting():
    lens = optics.Lens(
        sequence="sRz-",
        s=torch.tensor([[10.0], [100.0]]),
        c=torch.tensor([[0.0], [-0.02]]),
        nd=torch.tensor([[1.5]]),
        vd=torch.tensor([[50.0]]),
        dpgf=torch.tensor([[0.0]]),
        a=torch.empty((0, 1, 0)),
        d=torch.empty((0, 1, 0)),
        m=torch.empty((0, 1, 0)),
        z=torch.tensor([[[[0.0, 0.0, 0.0, 1.0], [0.005, 0.0, 0.2, 2.0]]]]),
        w0=550.0,
    )

    refractive_profiles = []
    for surface_type, _, sag_fn, _ in lens.return_geometry():
        if surface_type == "r":
            refractive_profiles.append(sag_fn(torch.tensor([0.5, 1.5])))

    torch.testing.assert_close(refractive_profiles[0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(refractive_profiles[1], torch.tensor([-0.0025, 0.18875]))


def test_parameterization_scales_zones_and_freezes_topology():
    zones = [
        [
            [0.0, 1e-6, 0.0, 5.0],
            [1e-3, 2e-6, 1.0, 10.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ]
    parameterization = LensParameterization(
        lens_sequence="sRz-",
        s=[10.0, 100.0],
        nd=[1.5],
        vd=[50.0],
        dpgf=[0.0],
        c=[0.0, -0.01],
        a=[],
        d=[],
        m=[],
        z=zones,
        nominal_wavelength=550.0,
        scale_factor=10.0,
    )

    normalized = parameterization.lens_variables["z"]
    torch.testing.assert_close(
        normalized[0, 0, 1], torch.tensor([0.01, 0.002, 0.1, 1.0])
    )
    torch.testing.assert_close(parameterization.lens.z[:, 0], torch.tensor(zones))

    frozen = parameterization.freeze_dict["z"][0, 0]
    assert frozen[:, 3].all()
    assert frozen[0, 0] and frozen[0, 2]
    assert frozen[2].all()
    assert not frozen[1, :3].any()


def test_legacy_parameterization_without_z_remains_valid():
    parameterization = LensParameterization(
        lens_sequence="sR-",
        s=[10.0, 100.0],
        nd=[1.5],
        vd=[50.0],
        dpgf=[0.0],
        c=[0.0, -0.01],
        a=[],
        d=[],
        m=[],
        nominal_wavelength=550.0,
    )

    assert parameterization.lens.z.shape == (0, 1, 0, 4)
