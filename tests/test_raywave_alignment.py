"""Alignment of the ray-wave path against the RayWave / MATLAB reference.

add by cjy.

Reference: ``deeplens.raywave`` in the AutoLens project, configuration
``configs/raywave/annule_offaxis_inline.json``.  That implementation is itself
validated against MATLAB (spectral Strehl to 1e-13, exit-pupil OPD RMS to
1e-12 waves) and against Zemax (single-ray trace to 1e-16), see its
``deeplens/raywave/VALIDATION.md``.

Prescription: flat fused-silica entrance (thickness 4 mm) followed by a
three-zone annular exit surface (28 mm to the image), 6 mm entrance pupil,
1.5 deg field, 20 C.  The zones are C0-continuous -- ``Zoff`` restores
continuity across the differing per-zone ``A2`` -- so this locks intersection,
refraction, optical-path accumulation and the diffraction integral, but NOT
the stepped-Fresnel branch physics.

This is an IMPLEMENTATION-EQUIVALENCE gate, not a physical-accuracy gate: it
feeds both codes the same pupil samples, so quadrature error cancels.  The
reference values themselves carry roughly 1% quadrature error -- re-running the
reference at ob_m = 129 -> 513 moves S(750 nm) from 0.013607 to 0.013738, and a
dense native sweep lands at 0.013742.  Numerical convergence is covered
separately; see the sampling table in the README.

Conventions taken from the reference (they matter at the 1e-4 level off axis):
  * the entrance pupil is a Cartesian grid masked to a circle of radius D / 2;
  * ``spectral_SR`` is sampled at the chief ray, not at the ray centroid;
  * the ideal reference sphere converges on the optical AXIS, not the chief ray.
"""

import math

import pytest
import torch

from eisoptx.modeling import optics
from eisoptx.modeling import ray_analysis as ra
from eisoptx.modeling.simulation import PSFSampler

WAVELENGTHS_NM = [550.0, 650.0, 750.0]
DESIGN_INDEX = 1
FIELD_DEG = 1.5
PUPIL_DIAMETER_MM = 6.0
PUPIL_SAMPLING = 129
THICKNESS_MM = 4.0
IMAGE_DISTANCE_MM = 28.0

# Fused silica at 20 C, from the reference thermal model.
FUSED_SILICA_INDEX = [1.4599108862207382, 1.4565349734639566, 1.4542367368990914]
# Hartmann parameters that reproduce those three indices exactly.
HARTMANN_ND_VD_DPGF = (1.4584708813156901, 68.64967602803766, -0.02414519374000078)

# [delta_A1, A2, delta_Z, Rmax]; the base curvature is zero, so delta_A1 == A1.
ZONES = [
    [-0.04, 0.00012, 0.0, 1.2],
    [-0.04, 0.00013, -2.0736e-05, 2.4],
    [-0.04, 0.00014, -0.00035251776, 5.0],
]

REFERENCE_STREHL = [
    0.0015976909186419118,
    0.006631789918667075,
    0.01360720701305359,
]

# (x_entrance, y_entrance, x_exit_pupil, y_exit_pupil, optical_path) at 650 nm,
# every 857th ray of the masked pupil grid.
REFERENCE_RAYS = [
    (
        2.220446049250313e-16,
        -3.0,
        2.194298273215118e-16,
        -2.8907327199221133,
        5.598637386008143,
    ),
    (0.796875, -2.296875, 0.7905878600688676, -2.2055496863271427, 5.667764564128487),
    (2.203125, -1.875, 2.1786088217150095, -1.7803210501583246, 5.6358256582614406),
    (-2.390625, -1.453125, -2.365535876525426, -1.364182036230247, 5.654958869228734),
    (-0.140625, -1.125, -0.14039368206139166, -1.0509943838324365, 5.777141645717943),
    (0.234375, -0.796875, 0.23417626334851796, -0.7241688281047268, 5.7956552276848905),
    (-0.609375, -0.46875, -0.6089048199953765, -0.396369698570322, 5.80518103493957),
    (-2.109375, -0.140625, -2.096106435270024, -0.06681746035844821, 5.743752357019757),
    (2.25, 0.140625, 2.2338760492345755, 0.21269320025556834, 5.739688992260124),
    (0.75, 0.46875, 0.7490706276989, 0.5402605560085786, 5.823799028341869),
    (-0.09375, 0.796875, -0.09364631527911325, 0.868064297433115, 5.834039801145901),
    (0.28125, 1.125, 0.28063875687291695, 1.1947941582104389, 5.829095480503013),
    (2.53125, 1.453125, 2.501503754973777, 1.5099868320991572, 5.712018555915922),
    (-2.0625, 1.875, -2.040068780619604, 1.9283759837876424, 5.73461070865086),
    (-0.65625, 2.296875, -0.6507988590983078, 2.351082417088332, 5.779788635851935),
]


def _build_lens():
    nd, vd, dpgf = HARTMANN_ND_VD_DPGF
    kw = {"dtype": torch.float64}
    return optics.Lens(
        sequence="sRz-",
        s=torch.tensor([[THICKNESS_MM], [IMAGE_DISTANCE_MM]], **kw),
        c=torch.zeros(2, 1, **kw),
        nd=torch.tensor([[nd]], **kw),
        vd=torch.tensor([[vd]], **kw),
        dpgf=torch.tensor([[dpgf]], **kw),
        a=torch.empty((0, 1, 0), **kw),
        d=torch.empty((0, 1, 0), **kw),
        m=torch.empty((0, 1, 0), **kw),
        z=torch.tensor(ZONES, **kw).reshape(1, 1, 3, 4),
        w0=WAVELENGTHS_NM[DESIGN_INDEX],
    )


def _masked_pupil():
    """Reference entrance grid: Cartesian, zeros nudged by eps, circular mask."""
    half = PUPIL_DIAMETER_MM / 2
    axis = torch.linspace(-half, half, PUPIL_SAMPLING, dtype=torch.float64)
    x, y = torch.meshgrid(axis, axis, indexing="xy")
    eps = torch.finfo(torch.float64).eps
    x = x.where(x != 0, torch.full_like(x, eps))
    y = y.where(y != 0, torch.full_like(y, eps))
    keep = (x**2 + y**2) <= half**2
    return x[keep], y[keep]


def _trace(lens, x, y):
    """Trace a collimated bundle at FIELD_DEG for every wavelength at once."""
    n_rays, n_wl = x.numel(), len(WAVELENGTHS_NM)
    theta = math.radians(FIELD_DEG)
    r0 = torch.zeros(3, n_rays, n_wl, 1, dtype=torch.float64)
    r0[0, :, :, 0] = x[:, None]
    r0[1, :, :, 0] = y[:, None]
    d0 = torch.zeros(3, n_rays, n_wl, 1, dtype=torch.float64)
    d0[1], d0[2] = math.sin(theta), math.cos(theta)
    return list(lens.trace_rays(r0, d0, WAVELENGTHS_NM, yield_on="end"))[-1]


def _exit_pupil(lens, x, y):
    """Back-project to the last-surface vertex plane, as forward_ray_wave does."""
    r, d, status, info = _trace(lens, x, y)
    back = lens.s[-1] / d[2]
    return (
        r[0] - d[0] * back,
        r[1] - d[1] * back,
        info["opl"] - back,
        status,
        r,
        d,
    )


def test_hartmann_reproduces_fused_silica():
    nd, vd, dpgf = HARTMANN_ND_VD_DPGF
    index = optics.hartmann_dispersion(
        torch.tensor(WAVELENGTHS_NM, dtype=torch.float64),
        torch.tensor([nd], dtype=torch.float64),
        torch.tensor([vd], dtype=torch.float64),
        torch.tensor([dpgf], dtype=torch.float64),
    ).reshape(-1)
    torch.testing.assert_close(
        index,
        torch.tensor(FUSED_SILICA_INDEX, dtype=torch.float64),
        atol=1e-13,
        rtol=0,
    )


def test_exit_pupil_optical_path_matches_raywave():
    lens = _build_lens()
    sample = torch.tensor(REFERENCE_RAYS, dtype=torch.float64)
    xe, ye = sample[:, 0], sample[:, 1]
    xp, yp, opl, status, *_ = _exit_pupil(lens, xe, ye)

    assert (status == 0).all()
    wi = DESIGN_INDEX
    torch.testing.assert_close(xp[:, wi, 0], sample[:, 2], atol=1e-12, rtol=0)
    torch.testing.assert_close(yp[:, wi, 0], sample[:, 3], atol=1e-12, rtol=0)
    # Optical path is absolute here: the reference uses the same incident
    # eikonal convention (y * sin(theta)), so there is no piston offset.
    wavelength_mm = WAVELENGTHS_NM[wi] * 1e-6
    error_waves = (opl[:, wi, 0] - sample[:, 4]).abs().max() / wavelength_mm
    assert error_waves < 1e-9, f"exit-pupil OPL differs by {error_waves} waves"


@pytest.mark.parametrize("wavelength_index", range(len(WAVELENGTHS_NM)))
def test_offaxis_annular_strehl_matches_raywave(wavelength_index):
    """End-to-end Strehl at the chief ray, in the reference's convention."""
    lens = _build_lens()
    x, y = _masked_pupil()
    xp, yp, opl, status, *_ = _exit_pupil(lens, x, y)
    assert (status == 0).all()

    eps = torch.finfo(torch.float64).eps
    chief = torch.tensor([eps], dtype=torch.float64)
    _, _, _, _, r_chief, _ = _exit_pupil(lens, chief, chief)

    wi = wavelength_index
    xc, yc = r_chief[0, 0, wi, 0], r_chief[1, 0, wi, 0]
    xp, yp, opl = xp[:, wi, 0], yp[:, wi, 0], opl[:, wi, 0]
    distance = lens.s[-1]

    to_chief = ((xp - xc) ** 2 + (yp - yc) ** 2 + distance**2).sqrt()
    wavenumber = 2 * math.pi / (WAVELENGTHS_NM[wi] * 1e-6)
    field = (
        (to_chief + distance)
        / (2 * to_chief)
        * torch.exp(1j * wavenumber * (opl + to_chief))
    ).sum()
    # Ideal reference sphere converging on the optical axis.
    to_axis = (xp**2 + yp**2 + distance**2).sqrt()
    ideal = ((to_axis + distance) / (2 * to_axis)).sum()

    strehl = (field.abs() ** 2 / ideal**2).item()
    reference = REFERENCE_STREHL[wi]
    assert abs(strehl - reference) / reference < 1e-9, (
        f"{WAVELENGTHS_NM[wi]} nm: {strehl} vs reference {reference}"
    )


def test_forward_ray_wave_reproduces_the_reference_normalization():
    """The shipped Kirchhoff path must match an independent sum at its own grid centre."""
    lens = _build_lens()
    x, y = _masked_pupil()
    r, d, status, info = _trace(lens, x, y)
    n_wl = len(WAVELENGTHS_NM)

    sampler = PSFSampler(
        psf_shape=(5, 5),
        psf_abs_size=(0.0035 * 5, 0.0035 * 5),
        wavelength_weights=[1.0] * n_wl,
        wavelengths=WAVELENGTHS_NM,
        psf_normalization="strehl",
    )
    psfs = sampler.forward_ray_wave(
        r[:, None],
        d[:, None],
        info["opl"][None],
        status[None],
        lens.s[-1],
        torch.tensor(WAVELENGTHS_NM, dtype=torch.float64),
    )

    # forward_ray_wave centres its grid on one spectrum-weighted centroid that
    # is SHARED by every wavelength, so lateral colour stays in the PSF.  Reuse
    # the same helper rather than re-deriving it, so the centring is locked too.
    ray_valid = status[None] == 0
    cx = ra.evaluate_mean_ray_height(
        r[0][None], ray_valid, (1, 2), sampler.wavelength_weights
    )[0, 0, 0, 0]
    cy = ra.evaluate_mean_ray_height(
        r[1][None], ray_valid, (1, 2), sampler.wavelength_weights
    )[0, 0, 0, 0]

    valid = status == 0
    back = lens.s[-1] / d[2]
    xp, yp = r[0] - d[0] * back, r[1] - d[1] * back
    opl = info["opl"] - back
    distance = lens.s[-1]
    for wi in range(n_wl):
        to_centre = (
            (xp[:, wi, 0] - cx) ** 2 + (yp[:, wi, 0] - cy) ** 2 + distance**2
        ).sqrt()
        wavenumber = 2 * math.pi / (WAVELENGTHS_NM[wi] * 1e-6)
        field = (
            (to_centre + distance)
            / (2 * to_centre)
            * torch.exp(1j * wavenumber * (opl[:, wi, 0] + to_centre))
        ).sum()
        to_axis = (xp[:, wi, 0] ** 2 + yp[:, wi, 0] ** 2 + distance**2).sqrt()
        ideal = ((to_axis + distance) / (2 * to_axis)).sum()
        expected = (field.abs() ** 2 / ideal**2).item()
        got = psfs[0, 0, wi, 2, 2].item()
        assert valid[:, wi, 0].all()
        assert abs(got - expected) / expected < 1e-9, f"{WAVELENGTHS_NM[wi]} nm"
