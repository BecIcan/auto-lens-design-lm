import math
from pathlib import Path

import pytest
import torch
import yaml

from eisoptx.modeling import optics
from eisoptx.modeling.ray_initialization import RayInitialization, skew_uniform
from eisoptx.modeling.simulation import OpticsSimulator, PSFSampler
from eisoptx.optimization.parameterization import LensParameterization
from eisoptx.optimization.residuals import (
    CoherentWavefrontResiduals,
    HDOEBranchConstraintResiduals,
    HDOEBranchResiduals,
    HDOEPhaseResiduals,
)


DTYPE = torch.float64


@pytest.mark.parametrize("order", [0, -1, 1.5])
def test_hdoe_residual_requires_positive_integer_order(order):
    with pytest.raises(ValueError, match="positive integer"):
        HDOEPhaseResiduals(
            weight=1.0,
            diffraction_order=order,
            design_wavelength=550.0,
        )


def empty_surface_parameters(n_surfaces=0, n_parameters=0):
    return torch.empty((n_surfaces, 1, n_parameters), dtype=DTYPE)


def make_lens(sequence, s, c=(), nd=(), vd=(), dpgf=(), d=(), z=None):
    return optics.Lens(
        sequence=sequence,
        s=torch.tensor(s, dtype=DTYPE).view(-1, 1),
        c=torch.tensor(c, dtype=DTYPE).view(-1, 1),
        nd=torch.tensor(nd, dtype=DTYPE).view(-1, 1),
        vd=torch.tensor(vd, dtype=DTYPE).view(-1, 1),
        dpgf=torch.tensor(dpgf, dtype=DTYPE).view(-1, 1),
        a=empty_surface_parameters(),
        d=(
            torch.tensor(d, dtype=DTYPE).view(1, 1, -1)
            if len(d) > 0
            else empty_surface_parameters()
        ),
        m=empty_surface_parameters(),
        z=(torch.tensor(z, dtype=DTYPE).view(1, 1, -1, 4) if z else None),
        w0=550.0,
    )


def axial_ray(n_wavelengths=1):
    r = torch.zeros((3, 1, n_wavelengths, 1), dtype=DTYPE)
    d = torch.zeros_like(r)
    d[2] = 1
    return r, d


def test_opl_tracks_dispersive_glass_and_air_distances():
    wavelengths = torch.tensor([486.1, 656.3], dtype=DTYPE)
    lens = make_lens(
        "sR-",
        s=(2.0, 3.0),
        c=(0.0, 0.0),
        nd=(1.5,),
        vd=(50.0,),
        dpgf=(0.0,),
    )
    r, d = axial_ray(len(wavelengths))

    *_, event_info = next(lens.trace_rays(r, d, wavelengths, yield_on="end"))

    refractive_index = optics.hartmann_dispersion(
        wavelengths, lens.nd[0], lens.vd[0], lens.dpgf[0]
    )[0]
    expected = 2.0 * refractive_index + 3.0
    torch.testing.assert_close(event_info["opl"].flatten(), expected)


def test_opl_preserves_incident_plane_wave_tilt():
    lens = make_lens("s-", s=(5.0,))
    theta = torch.deg2rad(torch.tensor(12.0, dtype=DTYPE))
    r = torch.zeros((3, 2, 1, 1), dtype=DTYPE)
    r[1, 1] = 1.25
    d = torch.zeros_like(r)
    d[1] = theta.sin()
    d[2] = theta.cos()

    *_, event_info = next(lens.trace_rays(r, d, [550.0], yield_on="end"))

    opl = event_info["opl"].flatten()
    expected_difference = 1.25 * theta.sin()
    torch.testing.assert_close(opl[1] - opl[0], expected_difference)


def test_zonal_relief_changes_opl_through_physical_path_only():
    wavelength = torch.tensor([550.0], dtype=DTYPE)
    base = make_lens(
        "sRz-",
        s=(2.0, 3.0),
        c=(0.0, 0.0),
        nd=(1.5,),
        vd=(50.0,),
        dpgf=(0.0,),
        z=[[[0.0, 0.0, 0.0, 5.0]]],
    )
    relief_height = 0.1
    relief = make_lens(
        "sRz-",
        s=(2.0, 3.0),
        c=(0.0, 0.0),
        nd=(1.5,),
        vd=(50.0,),
        dpgf=(0.0,),
        z=[[[0.0, 0.0, relief_height, 5.0]]],
    )
    r, d = axial_ray()

    base_opl = next(base.trace_rays(r, d, wavelength, yield_on="end"))[3]["opl"]
    relief_opl = next(relief.trace_rays(r, d, wavelength, yield_on="end"))[3]["opl"]
    refractive_index = optics.hartmann_dispersion(
        wavelength, relief.nd[0], relief.vd[0], relief.dpgf[0]
    )[0, 0]

    expected_difference = (refractive_index - 1) * relief_height
    torch.testing.assert_close((relief_opl - base_opl).squeeze(), expected_difference)


def test_diffractive_event_reports_equivalent_phase_opd():
    lens = make_lens("s-d-", s=(1.0, 2.0), d=(0.1, -0.02))
    r, direction = axial_ray()
    r[0] = 0.5

    traced_events = list(lens.trace_rays(r, direction, [550.0], yield_on="all"))
    event_info = next(
        info
        for event, (*_, info) in zip(lens.sequence.events, traced_events)
        if event["type"] == "d"
    )

    rho = 0.5**2
    expected = 0.1 * rho - 0.02 * rho**2
    torch.testing.assert_close(
        event_info["phase_opd"].squeeze(), torch.tensor(expected, dtype=DTYPE)
    )
    torch.testing.assert_close(
        event_info["opl"].squeeze(), torch.tensor(1.0 + expected, dtype=DTYPE)
    )


def test_hdoe_phase_residual_removes_integer_zone_branches():
    diffraction_order = 480
    wavelength_nm = 550.0
    period = diffraction_order * wavelength_nm * 1e-6
    zone_index = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    final_opl = 123.4 + zone_index.to(DTYPE) * period
    opl = torch.stack((torch.zeros_like(final_opl), final_opl), dim=-1).view(
        1, 4, 1, 1, 2
    )
    zonal_index = zone_index.view(1, 4, 1, 1, 1)
    xy = torch.zeros((2, 1, 4, 1, 1), dtype=DTYPE)
    direction = torch.zeros((3, 1, 4, 1, 1), dtype=DTYPE)
    direction[2] = 1
    image_distance = torch.tensor([1.0], dtype=DTYPE)
    residual = HDOEPhaseResiduals(
        weight=1.0,
        diffraction_order=diffraction_order,
        design_wavelength=wavelength_nm,
    )

    torch.testing.assert_close(
        residual(
            opl,
            zonal_index,
            xy,
            direction,
            image_distance,
            [wavelength_nm],
        ),
        torch.zeros(4, dtype=DTYPE),
        atol=1e-14,
        rtol=0,
    )

    perturbed_opl = opl.clone().requires_grad_()
    perturbed_opl.data[0, 2, 0, 0, -1] += 1e-3
    loss = (
        residual(
            perturbed_opl,
            zonal_index,
            xy,
            direction,
            image_distance,
            [wavelength_nm],
        )
        .square()
        .sum()
    )
    gradient = torch.autograd.grad(loss, perturbed_opl)[0]
    assert loss > 0
    assert torch.isfinite(gradient).all()


def test_hdoe_branch_constraint_uses_adjacent_complete_opl_means():
    diffraction_order = 30
    wavelength_nm = 550.0
    period = diffraction_order * wavelength_nm * 1e-6
    zone_index = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    intra_zone = torch.tensor([-2e-6, 2e-6] * 3, dtype=DTYPE)
    final_opl = 20.0 + zone_index.to(DTYPE) * period + intra_zone
    opl = torch.stack((torch.zeros_like(final_opl), final_opl), dim=-1).view(
        1, 6, 1, 1, 2
    )
    zonal_index = zone_index.view(1, 6, 1, 1, 1)
    xy = torch.zeros((2, 1, 6, 1, 1), dtype=DTYPE)
    direction = torch.zeros((3, 1, 6, 1, 1), dtype=DTYPE)
    direction[2] = 1
    residual = HDOEBranchConstraintResiduals(
        weight=1.0,
        diffraction_order=diffraction_order,
        design_wavelength=wavelength_nm,
        n_zones=3,
    )

    value = residual(
        opl,
        zonal_index,
        xy,
        direction,
        torch.tensor([1.0], dtype=DTYPE),
        [wavelength_nm],
        torch.ones(6, dtype=DTYPE),
    )
    torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-14, rtol=0)

    perturbed = opl.clone().requires_grad_()
    perturbed.data[0, 4:, 0, 0, -1] += 1e-4
    perturbed_value = residual(
        perturbed,
        zonal_index,
        xy,
        direction,
        torch.tensor([1.0], dtype=DTYPE),
        [wavelength_nm],
        torch.ones(6, dtype=DTYPE),
    )
    jacobian = torch.autograd.functional.jacobian(
        lambda value: residual(
            value,
            zonal_index,
            xy,
            direction,
            torch.tensor([1.0], dtype=DTYPE),
            [wavelength_nm],
            torch.ones(6, dtype=DTYPE),
        ),
        perturbed,
    )
    torch.testing.assert_close(
        perturbed_value,
        torch.tensor([[[0.0, 1e-4]]], dtype=DTYPE),
        atol=1e-14,
        rtol=0,
    )
    assert torch.isfinite(jacobian).all()


def test_hdoe_branch_soft_residual_is_not_a_kkt_constraint():
    assert HDOEBranchConstraintResiduals.constraint is True
    assert HDOEBranchResiduals.constraint is False


def coherent_inputs(final_opl, wavelengths):
    final_opl = torch.as_tensor(final_opl, dtype=DTYPE)
    n_rays, n_wavelengths = final_opl.shape
    opl = torch.stack((torch.zeros_like(final_opl), final_opl), dim=-1).view(
        1, n_rays, n_wavelengths, 1, 2
    )
    xy = torch.zeros((2, 1, n_rays, n_wavelengths, 1), dtype=DTYPE)
    direction = torch.zeros((3, 1, n_rays, n_wavelengths, 1), dtype=DTYPE)
    direction[2] = 1
    return opl, xy, direction, torch.tensor([1.0], dtype=DTYPE), wavelengths


def test_coherent_wavefront_residual_is_piston_invariant_and_branch_continuous():
    wavelengths = [500.0, 600.0]
    wavelength_mm = torch.tensor(wavelengths, dtype=DTYPE) * 1e-6
    integer_branches = torch.tensor([0.0, 1.0, 3.0], dtype=DTYPE)[:, None]
    final_opl = 42.0 + integer_branches * wavelength_mm
    inputs = coherent_inputs(final_opl, wavelengths)
    residual = CoherentWavefrontResiduals(weight=1.0)

    value = residual(*inputs)
    shifted_inputs = coherent_inputs(final_opl + 17.25, wavelengths)
    shifted = residual(*shifted_inputs)

    torch.testing.assert_close(value, torch.zeros_like(value), atol=2e-10, rtol=0)
    torch.testing.assert_close(shifted, value, atol=2e-10, rtol=0)


def test_coherent_wavefront_residual_norm_equals_one_minus_coherent_fraction():
    wavelength_nm = 550.0
    wavelength_mm = wavelength_nm * 1e-6
    # Two equally weighted rays separated by pi have zero coherent sum.
    final_opl = torch.tensor([[0.0], [0.5 * wavelength_mm]], dtype=DTYPE)
    inputs = coherent_inputs(final_opl, [wavelength_nm])
    residual = CoherentWavefrontResiduals(weight=1.0)

    value = residual(*inputs, pupil_weights=torch.tensor([1.0, 1.0]))

    torch.testing.assert_close(value.square().sum(), torch.tensor(1.0, dtype=DTYPE))


def test_coherent_wavefront_residual_can_override_global_spectral_weights():
    wavelength_nm = [500.0, 600.0]
    wavelength_mm = torch.tensor(wavelength_nm, dtype=DTYPE) * 1e-6
    final_opl = torch.stack(
        (
            torch.tensor([0.0, 0.5], dtype=DTYPE) * wavelength_mm[0],
            torch.zeros(2, dtype=DTYPE),
        ),
        dim=1,
    )
    inputs = coherent_inputs(final_opl, wavelength_nm)
    residual = CoherentWavefrontResiduals(
        weight=1.0, wavelength_weights=[1.0, 0.0]
    )

    value = residual(
        *inputs,
        wavelength_weights=torch.tensor([0.0, 1.0], dtype=DTYPE),
    )

    torch.testing.assert_close(value.square().sum(), torch.tensor(1.0, dtype=DTYPE))


def test_coherent_wavefront_residual_uses_weights_for_reference_and_merit():
    wavelength_nm = 550.0
    final_opl = torch.tensor([[0.0], [0.17e-3]], dtype=DTYPE)
    weighted_inputs = list(coherent_inputs(final_opl, [wavelength_nm]))
    weighted_inputs[1][0, 0, :, 0, 0] = torch.tensor([0.0, 2.0], dtype=DTYPE)

    expanded_opl = torch.tensor([[0.0], [0.17e-3], [0.17e-3], [0.17e-3]], dtype=DTYPE)
    expanded_inputs = list(coherent_inputs(expanded_opl, [wavelength_nm]))
    expanded_inputs[1][0, 0, :, 0, 0] = torch.tensor(
        [0.0, 2.0, 2.0, 2.0], dtype=DTYPE
    )
    residual = CoherentWavefrontResiduals(weight=1.0)

    weighted_loss = residual(
        *weighted_inputs, pupil_weights=torch.tensor([1.0, 3.0], dtype=DTYPE)
    ).square().sum()
    expanded_loss = residual(*expanded_inputs).square().sum()

    torch.testing.assert_close(weighted_loss, expanded_loss, atol=1e-13, rtol=1e-12)


def test_coherent_wavefront_residual_has_finite_opl_jacobian():
    wavelength_nm = 550.0
    final_opl = torch.tensor([[0.0], [0.23e-3], [0.41e-3]], dtype=DTYPE)
    opl, *other = coherent_inputs(final_opl, [wavelength_nm])
    opl.requires_grad_()
    residual = CoherentWavefrontResiduals(weight=1.0)

    jacobian = torch.autograd.functional.jacobian(
        lambda value: residual(value, *other).reshape(-1), opl
    )

    assert torch.isfinite(jacobian).all()
    assert jacobian.abs().max() > 0


def test_coherent_scalar_and_vector_have_same_objective_and_gradient():
    wavelengths = [486.1, 550.0, 656.3]
    final_opl = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.13e-3, 0.21e-3, 0.34e-3],
            [0.29e-3, 0.37e-3, 0.48e-3],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )
    vector = CoherentWavefrontResiduals(weight=1.0, reduce_to_scalar=False)
    scalar = CoherentWavefrontResiduals(weight=1.0, reduce_to_scalar=True)

    def objective(residual):
        inputs = coherent_inputs(final_opl, wavelengths)
        value = residual(*inputs)
        return 0.5 * value.square().sum(), value

    vector_objective, vector_value = objective(vector)
    vector_gradient = torch.autograd.grad(vector_objective, final_opl)[0]
    scalar_objective, scalar_value = objective(scalar)
    scalar_gradient = torch.autograd.grad(scalar_objective, final_opl)[0]

    assert vector_value.numel() > scalar_value.numel() == 1
    torch.testing.assert_close(scalar_objective, vector_objective, atol=1e-13, rtol=1e-12)
    torch.testing.assert_close(scalar_gradient, vector_gradient, atol=1e-9, rtol=1e-10)


def ideal_focusing_wavefront(n_r=24, n_theta=24, diameter=5.0, focal_length=10.0):
    x, y = skew_uniform(n_r, n_theta)
    x = x.to(DTYPE) * diameter / 2
    y = y.to(DTYPE) * diameter / 2
    distance = (x**2 + y**2 + focal_length**2).sqrt()
    direction = torch.stack((-x / distance, -y / distance, focal_length / distance))[
        :, None, :, None, None
    ]
    r_image = torch.zeros_like(direction)
    r_image[2] = focal_length
    opl_image = torch.zeros((1, x.numel(), 1, 1), dtype=DTYPE)
    ray_status = torch.zeros_like(opl_image, dtype=torch.int)
    return r_image, direction, opl_image, ray_status


def test_ray_wave_psf_is_piston_invariant_and_resolves_airy_first_zero():
    focal_length = 10.0
    diameter = 5.0
    wavelength_nm = 550.0
    psf_shape = 49
    psf_abs_size = 0.008
    sampler = PSFSampler(
        psf_shape,
        psf_abs_size,
        [1.0],
        wave_chunk_size=128,
    ).double()
    r_image, direction, opl_image, ray_status = ideal_focusing_wavefront(
        diameter=diameter, focal_length=focal_length
    )
    wavelengths = torch.tensor([wavelength_nm], dtype=DTYPE)
    reference_distance = torch.tensor([focal_length], dtype=DTYPE)

    psf = sampler.forward_ray_wave(
        r_image,
        direction,
        opl_image,
        ray_status,
        reference_distance,
        wavelengths,
    )
    piston_psf = sampler.forward_ray_wave(
        r_image,
        direction,
        opl_image + 3.14159,
        ray_status,
        reference_distance,
        wavelengths,
    )

    torch.testing.assert_close(psf.sum(), torch.tensor(1.0, dtype=DTYPE))
    torch.testing.assert_close(psf, piston_psf, rtol=1e-10, atol=1e-12)

    first_zero = 1.22 * wavelength_nm * 1e-6 * focal_length / diameter
    first_zero_index = round(first_zero / (psf_abs_size / psf_shape))
    center = psf_shape // 2
    first_zero_intensity = psf[0, 0, 0, center, center + first_zero_index]
    peak_intensity = psf[0, 0, 0, center, center]
    assert first_zero_intensity / peak_intensity < 5e-3


def test_ray_wave_psf_has_finite_local_gradient():
    sampler = PSFSampler(17, 0.006, [1.0], wave_chunk_size=64).double()
    r_image, direction, opl_image, ray_status = ideal_focusing_wavefront(
        n_r=8, n_theta=8
    )
    defocus = torch.tensor(0.01, dtype=DTYPE, requires_grad=True)
    psf = sampler.forward_ray_wave(
        r_image,
        direction,
        opl_image,
        ray_status,
        torch.tensor([10.0], dtype=DTYPE) + defocus,
        torch.tensor([550.0], dtype=DTYPE),
    )

    center_intensity = psf[..., 8, 8].sum()
    gradient = torch.autograd.grad(center_intensity, defocus)[0]
    assert torch.isfinite(gradient)
    assert not math.isclose(gradient.item(), 0.0, abs_tol=1e-12)


def test_visible_annular_seed_is_diffraction_limited_on_axis():
    design_file = (
        Path(__file__).parents[1]
        / "configs"
        / "raywave_zonal"
        / "designs"
        / "visible_f100_f2_m480.yml"
    )
    with design_file.open(encoding="utf-8") as stream:
        parameterization_args = yaml.safe_load(stream)["model"][
            "lens_parameterization"
        ]["init_args"]
    lens = LensParameterization(**parameterization_args).double().lens
    ray_initialization = RayInitialization(
        aperture=50.0,
        aperture_type="epd",
        hfov=0.0,
        n_fields=1,
        wavelengths=[550.0],
        pupil_sampling_mode="skew_uniform",
        pupil_sampling_kwargs={"n_r": 32, "n_theta": 32},
    )
    simulator = OpticsSimulator(
        shape=(65, 65),
        shape_type="psf",
        psf_abs_size=0.01,
        sensor_diagonal=(100.0, 3.0),
        psf_grid_shape=(1, 1),
        wavelength_weights=([1.0], [1.0], [1.0]),
        patch_overlap=0.25,
        wavelengths=[550.0],
        psf_mode="ray_wave",
        wave_chunk_size=256,
    ).double()

    actual_psf = simulator.build_optics_model(lens, ray_initialization)[0][0, 0, 0]

    x, y = skew_uniform(32, 32)
    x = x.to(DTYPE) * 25.0
    y = y.to(DTYPE) * 25.0
    focal_length = lens.s[-1, 0]
    distance = (x**2 + y**2 + focal_length**2).sqrt()
    ideal_direction = torch.stack(
        (-x / distance, -y / distance, focal_length.expand_as(x) / distance)
    )[:, None, :, None, None]
    ideal_r_image = torch.zeros_like(ideal_direction)
    ideal_r_image[2] = focal_length
    ideal_opl_image = torch.zeros((1, x.numel(), 1, 1), dtype=DTYPE)
    ideal_status = torch.zeros_like(ideal_opl_image, dtype=torch.int)
    ideal_psf = simulator.psf_sampler.forward_ray_wave(
        ideal_r_image,
        ideal_direction,
        ideal_opl_image,
        ideal_status,
        focal_length.view(1),
        torch.tensor([550.0], dtype=DTYPE),
    )[0, 0, 0]

    normalized_peak_ratio = actual_psf.max() / ideal_psf.max()
    assert normalized_peak_ratio > 0.99
